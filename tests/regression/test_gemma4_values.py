# Copyright (c) 2025-2026 Jaegeun Han
#
# SPDX-License-Identifier: Apache-2.0

"""Fixed-seed forward and gradient parity against Google's Transformers decoder."""

import pytest
import torch
from tests.fixtures.gemma4 import gemma4_pair

from ironcore.checkpointing.weight_mapping import Architecture, WeightMapper
from ironcore.layers.gemma4 import Gemma4RMSNorm


@pytest.mark.parametrize("variant", ["E2B", "E4B", "31B", "A4B"])
def test_gemma4_forward_and_backward_values(monkeypatch, variant):
    torch.manual_seed(42)
    native, reference, _ = gemma4_pair(monkeypatch, variant)
    native.train()
    reference.train()
    # Extends beyond the local window, so a full-only mask cannot pass parity.
    tokens = torch.tensor([[2, 3, 4, 5, 6, 7], [2, 8, 9, 10, 11, 12]])
    actual, _ = native(tokens)
    expected = reference(tokens, use_cache=False).logits
    torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-5)
    coefficients = torch.linspace(-0.7, 0.9, actual.numel()).reshape_as(actual)
    (actual * coefficients).mean().backward()
    (expected * coefficients).mean().backward()
    gradients = {name: param.grad for name, param in reference.named_parameters()}
    mapped = WeightMapper(Architecture.GEMMA4, 4).hf_to_ironcore(gradients)
    for name, param in native.named_parameters():
        assert param.grad is not None, name
        torch.testing.assert_close(
            param.grad,
            mapped[name],
            atol=5e-5,
            rtol=5e-4,
            msg=lambda info, key=name: f"{key}: {info}",
        )


def test_gemma4_rmsnorm_forward_and_backward_values():
    torch.manual_seed(42)
    norm = Gemma4RMSNorm(4, eps=1e-6)
    with torch.no_grad():
        norm.weight.copy_(torch.tensor([0.5, 1.0, 1.5, 2.0]))
    x = torch.randn(2, 4, requires_grad=True)
    other = x.detach().clone().requires_grad_(True)
    scale = norm.weight.detach().clone().requires_grad_(True)
    expected = other * (other.square().mean(-1, keepdim=True) + 1e-6).pow(-0.5) * scale
    actual = norm(x)
    torch.testing.assert_close(actual, expected)
    actual.square().sum().backward()
    expected.square().sum().backward()
    torch.testing.assert_close(x.grad, other.grad)
    torch.testing.assert_close(norm.weight.grad, scale.grad)


def test_gemma4_per_layer_embedding_padding_forward_and_backward(monkeypatch):
    torch.manual_seed(42)
    native, reference, _ = gemma4_pair(monkeypatch)
    tokens = torch.tensor([[0, 2, 18, 0, 31]])
    actual = native.model.embed_tokens_per_layer(tokens)
    expected = torch.nn.functional.embedding(
        tokens, reference.model.embed_tokens_per_layer.weight, padding_idx=0
    )
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    weights = torch.linspace(-0.7, 0.9, actual.numel()).reshape_as(actual)
    (actual * weights).sum().backward()
    (expected * weights).sum().backward()
    torch.testing.assert_close(
        native.model.embed_tokens_per_layer.weight.grad,
        reference.model.embed_tokens_per_layer.weight.grad,
        atol=0,
        rtol=0,
    )
    assert native.model.embed_tokens_per_layer.weight.grad[0].count_nonzero() == 0
