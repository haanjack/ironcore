# Copyright (c) 2025-2026 Jaegeun Han
# SPDX-License-Identifier: Apache-2.0
"""Imported Llama checkpoints retain the reference norm rounding under AMP."""

import pytest
import torch
from tests.fixtures.config_fixtures import create_test_config

from ironcore.layers.layernorm.fused_rms_norm import RmsNorm


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_llama_rmsnorm_matches_hf_forward_and_backward(dtype, device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA AMP regression requires a GPU")
    from transformers.models.llama.modeling_llama import LlamaRMSNorm

    config = create_test_config(d_model=16)
    config.model.hf_model_type = "llama"
    config.model.ln_eps = 1e-5
    native = RmsNorm(config).to(device=device, dtype=dtype)
    reference = LlamaRMSNorm(16, eps=1e-5).to(device=device, dtype=dtype)
    generator = torch.Generator().manual_seed(71)
    weight = torch.rand(16, generator=generator) + 0.5
    with torch.no_grad():
        native.layernorm.weight.copy_(weight)
        reference.weight.copy_(weight)
    x = torch.randn(3, 7, 16, generator=generator).to(device=device, dtype=dtype)
    x.requires_grad_()
    other = x.detach().clone().requires_grad_()
    with torch.autocast(device, dtype=torch.bfloat16, enabled=dtype == torch.bfloat16):
        actual, expected = native(x), reference(other)
        coefficients = torch.randn(3, 7, 16, generator=generator).to(device=device, dtype=dtype)
        (actual * coefficients).float().sum().backward()
        (expected * coefficients).float().sum().backward()
    assert actual.dtype == expected.dtype == dtype
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(x.grad, other.grad, rtol=0, atol=0)
    torch.testing.assert_close(native.layernorm.weight.grad, reference.weight.grad, rtol=0, atol=0)
