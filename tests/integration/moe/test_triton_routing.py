# Copyright (c) 2025-2026 Jaegeun Han
# SPDX-License-Identifier: Apache-2.0

"""Triton tails, padding, duplicate scatter and strided tensors against CUDA ATen."""

from __future__ import annotations

import pytest
import torch

pytestmark = [
    pytest.mark.cuda,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required"),
]


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("padded", [False, True])
def test_routing_kernel_values_and_strides(dtype, padded):
    from ironcore.layers.moe.scheduled import _Block, _pack
    from ironcore.layers.moe.triton_routing import (
        gather_gradient,
        scatter,
        store_weight_gradient,
        weighted_scatter,
    )

    torch.manual_seed(99)
    # Channel count 37 exercises masked tails; transpose makes source strided.
    hidden = torch.randn(37, 9, device="cuda", dtype=dtype).t()
    weights = torch.randn(18, device="cuda", dtype=dtype)
    order = torch.randperm(18, device="cuda")
    counts = torch.tensor([11, 7], device="cuda")
    prefixes = torch.tensor([0, 11], device="cuda")
    block = _Block((0, 1), 5, (11, 7), 4) if padded else _Block((0, 1), 5, (6, 7))
    expected = _pack(hidden, weights, order, prefixes, counts, block, 2, False)
    actual = _pack(hidden, weights, order, prefixes, counts, block, 2, True)
    for a, b in zip(actual, expected, strict=True):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    tokens, selected, assignments, ids = actual
    values = torch.randn(37, ids.numel(), device="cuda", dtype=dtype).t()
    reference = torch.zeros_like(hidden, dtype=torch.float32)
    valid = ids >= 0
    reference.index_add_(0, ids[valid], (values * selected[:, None])[valid].float())
    accumulated = torch.zeros_like(reference)
    weighted_scatter(accumulated, values, selected, ids)
    torch.testing.assert_close(accumulated, reference, atol=1e-6, rtol=1e-6)
    plain = torch.zeros_like(reference)
    scatter(plain, values, ids)
    reference.zero_().index_add_(0, ids[valid], values[valid].float())
    torch.testing.assert_close(plain, reference, atol=1e-6, rtol=1e-6)
    upstream = torch.randn(37, 9, device="cuda", dtype=dtype).t()
    gathered = gather_gradient(upstream, ids)
    torch.testing.assert_close(gathered, upstream[ids.clamp_min(0)].masked_fill(~valid[:, None], 0))
    source = selected.square()
    stored = torch.zeros_like(weights)
    store_weight_gradient(stored, source, assignments)
    reference_weights = torch.zeros_like(weights).index_copy_(0, assignments[valid], source[valid])
    torch.testing.assert_close(stored, reference_weights, atol=0, rtol=0)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_unique_scatter_avoids_atomics_and_matches_addition(dtype):
    from ironcore.layers.moe.triton_routing import scatter, weighted_scatter

    torch.manual_seed(77)
    ids = torch.tensor([0, 2, 5, 1, -1], device="cuda")
    values = torch.randn(37, 5, device="cuda", dtype=dtype).t()
    weights = torch.randn(5, device="cuda", dtype=dtype)
    initial = torch.randn(37, 6, device="cuda").t()
    actual = initial.clone()
    weighted_scatter(actual, values, weights, ids, unique=True)
    expected = initial.clone().index_add_(0, ids[:4], (values * weights[:, None])[:4].float())
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    plain = initial.clone()
    scatter(plain, values, ids, unique=True)
    expected = initial.clone().index_add_(0, ids[:4], values[:4].float())
    torch.testing.assert_close(plain, expected, atol=0, rtol=0)


def test_weight_gradient_store_crosses_tile_boundary():
    from ironcore.layers.moe.triton_routing import store_weight_gradient

    values = torch.arange(259, dtype=torch.float32, device="cuda")
    assignments = torch.arange(259, device="cuda") * 2
    assignments[-1] = -1
    actual = torch.zeros(518, device="cuda")
    store_weight_gradient(actual, values, assignments)
    expected = torch.zeros_like(actual).index_copy_(0, assignments[:-1], values[:-1])
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


def test_duplicate_expert_slots_keep_atomic_fallback():
    from copy import deepcopy

    from tests.fixtures.config_fixtures import create_moe_test_config
    from tests.fixtures.utils import single_gpu_env

    from ironcore.layers.moe import MoEMLP
    from ironcore.layers.moe.grouped import grouped_experts
    from ironcore.parallel import parallel_states as ps

    with single_gpu_env():
        ps.initialize_model_parallel(1, 2)
        try:
            torch.manual_seed(46)
            cfg = create_moe_test_config(
                hidden_size=32, intermediate_size=32, num_routed_experts=4, num_experts_per_token=2
            )
            cfg.model.activation_type = "swiglu"
            cfg.model.moe.expert_backend = "grouped"
            cfg.model.moe.grouped_token_budget = 4
            reference = MoEMLP(cfg).cuda()
            reference.init_weights()
            actual = deepcopy(reference)
            actual.config.model.moe.blockwise_backend = "triton"
            indices = torch.tensor([[[0, 0], [1, 2], [3, 3], [0, 2], [1, 1]]], device="cuda")
            x = torch.randn(1, 5, 32, device="cuda", requires_grad=True)
            y = x.detach().clone().requires_grad_()
            weights = torch.randn(1, 5, 2, device="cuda", requires_grad=True)
            new_weights = weights.detach().clone().requires_grad_()
            expected = grouped_experts(x, indices, weights, reference.routed_experts)
            result = grouped_experts(y, indices, new_weights, actual.routed_experts)
            torch.testing.assert_close(result, expected, atol=1e-6, rtol=3e-5)
            expected.square().sum().backward()
            result.square().sum().backward()
            pairs = [(y.grad, x.grad), (new_weights.grad, weights.grad)]
            for p, q in zip(
                actual.routed_experts.parameters(),
                reference.routed_experts.parameters(),
                strict=True,
            ):
                assert (p.grad is None) == (q.grad is None)
                if p.grad is not None:
                    pairs.append((p.grad, q.grad))
            for value, target in pairs:
                torch.testing.assert_close(value, target, atol=1e-6, rtol=3e-5)
        finally:
            ps.destroy_model_parallel()
