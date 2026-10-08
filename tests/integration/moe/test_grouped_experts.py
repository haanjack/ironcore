# Copyright (c) 2025-2026 Jaegeun Han
# SPDX-License-Identifier: Apache-2.0

"""Actual CUDA grouped GEMM values and owner/input derivatives against dense GEMMs."""

from copy import deepcopy

import pytest
import torch
from tests.fixtures.config_fixtures import create_moe_test_config
from tests.fixtures.utils import single_gpu_env

from ironcore.layers.moe import MoEMLP
from ironcore.parallel import parallel_states as ps

pytestmark = [
    pytest.mark.cuda,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required"),
]


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("idle", [False, True])
def test_native_grouped_expert_forward_and_backward(dtype, idle):
    with single_gpu_env():
        ps.initialize_model_parallel(1, 2)
        try:
            torch.manual_seed(52)
            config = create_moe_test_config(
                hidden_size=32,
                intermediate_size=64,
                num_shared_experts=1,
                num_routed_experts=4,
                num_experts_per_token=2,
                mlp_bias=True,
                dropout_mlp=0,
            )
            config.model.activation_type = "swiglu"
            reference = MoEMLP(config).cuda()
            reference.init_weights()
            with torch.no_grad():
                for name, parameter in reference.named_parameters():
                    if name.endswith("bias"):
                        parameter.normal_(0, 0.01)
            actual = deepcopy(reference)
            actual.expert_backend = "grouped"
            actual.config.model.moe.virtual_block_size = 3
            actual.config.model.moe.grouped_token_budget = 7
            if idle:
                for model in (reference, actual):
                    with torch.no_grad():
                        model.router.weight.zero_()
                        model.router.weight[:, 0] = 2
                        model.router.weight[:, 1] = 1
                        model.router.weight[:, 2] = -2
                        model.router.weight[:, 3] = -3
            x = torch.randn(2, 17, 32, device="cuda")
            if idle:
                x = x.abs()
            x.requires_grad_()
            y = x.detach().clone().requires_grad_()
            with torch.autocast("cuda", dtype=dtype, enabled=dtype != torch.float32):
                expected, result = reference(x), actual(y)
                a = expected.square().sum() + reference.get_aux_loss()
                b = result.square().sum() + actual.get_aux_loss()
            a.backward()
            b.backward()
            tolerance = (
                0.015 if dtype == torch.bfloat16 else 0.003 if dtype == torch.float16 else 3e-5
            )
            pairs = [(result, expected), (y.grad, x.grad)]
            for (name, p), (_, q) in zip(
                actual.named_parameters(), reference.named_parameters(), strict=True
            ):
                assert (p.grad is None) == (q.grad is None), name
                if p.grad is not None:
                    pairs.append((p.grad, q.grad))
            for value, target in pairs:
                torch.testing.assert_close(
                    value, target, atol=2e-6 if dtype == torch.float32 else 3e-4, rtol=tolerance
                )
                error = (value.float() - target.float()).norm()
                assert error <= tolerance * target.float().norm() + 3e-6
        finally:
            ps.destroy_model_parallel()
