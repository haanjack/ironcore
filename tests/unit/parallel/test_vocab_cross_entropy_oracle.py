# Copyright (c) 2025-2026 Jaegeun Han
# SPDX-License-Identifier: Apache-2.0
"""Independent stock CE oracle: ignored labels, dummy classes and shard VJPs."""

import pytest
import torch
import torch.nn.functional as F

from ironcore.parallel import parallel_states
from ironcore.parallel.tensor_parallel.cross_entropy import vocab_parallel_cross_entropy


@pytest.mark.parametrize("sharded", [False, True])
@pytest.mark.parametrize("rank", [0, 1])
def test_cross_entropy_matches_global_reference(monkeypatch, sharded, rank):
    torch.manual_seed(17)
    full = torch.randn(2, 3, 8, dtype=torch.float64)
    full[..., 7] = 99  # Dummy classes must not change the denominator/max.
    labels = torch.tensor([[0, 4, -100], [6, 7, 2]])
    targets = labels.masked_fill(labels >= 7, -100)
    cotangent = torch.randn(2, 3, dtype=torch.float64)
    reference = full.detach().clone().requires_grad_()
    expected = F.cross_entropy(
        reference[..., :7].reshape(-1, 7), targets.flatten(), reduction="none"
    ).view_as(labels)
    expected.backward(cotangent)
    size = 2 if sharded else 1
    rank = rank if sharded else 0
    monkeypatch.setattr(parallel_states, "get_tensor_model_parallel_world_size", lambda: size)
    monkeypatch.setattr(parallel_states, "get_tensor_model_parallel_rank", lambda: rank)
    monkeypatch.setattr(parallel_states, "get_tensor_model_parallel_group", lambda: None)
    if sharded:
        maximum = full[..., :7].max(-1).values
        shifted = full[..., :7] - maximum.unsqueeze(-1)
        predicted = shifted.gather(-1, labels.clamp(0, 6).unsqueeze(-1)).squeeze(-1)
        predicted = predicted.masked_fill((labels < 0) | (labels >= 7), 0)
        global_values = iter([maximum, predicted, shifted.exp().sum(-1)])

        def all_reduce(tensor, **kwargs):
            tensor.copy_(next(global_values))

        monkeypatch.setattr(torch.distributed, "all_reduce", all_reduce)
        native = full[..., rank * 4 : (rank + 1) * 4].clone().requires_grad_()
    else:
        native = full.clone().requires_grad_()
    actual = vocab_parallel_cross_entropy(native, labels, padding_start_idx=7)
    actual.backward(cotangent)
    gradient = reference.grad[..., rank * 4 : (rank + 1) * 4] if sharded else reference.grad
    torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(native.grad, gradient, rtol=1e-12, atol=1e-12)
