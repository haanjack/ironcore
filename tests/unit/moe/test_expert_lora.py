# Copyright (c) 2025-2026 Jaegeun Han
# SPDX-License-Identifier: Apache-2.0
"""Routed LoRA: independent loop reference, bounded backward, and idle experts."""

import copy

import pytest
import torch
from tests.fixtures.gemma4 import gemma4_pair

from ironcore.layers.moe.grouped import grouped_experts
from ironcore.parallel.random import reset_tensor_parallel_rng_tracker
from ironcore.peft import load_lora_adapter, merge_lora_weights, save_lora_adapter


@pytest.mark.parametrize(
    "targets", [["gate_proj"], ["up_proj", "down_proj"], ["gate_proj", "up_proj", "down_proj"]]
)
@pytest.mark.parametrize("backend", ["torch", "scheduled"])
def test_routed_lora_matches_loop_gradients_and_skips_idle_experts(monkeypatch, targets, backend):
    torch.manual_seed(17)
    native, _, _ = gemma4_pair(monkeypatch, "A4B", lora=True, lora_targets=targets)
    experts = native.model.layers[0].experts.double()
    with torch.no_grad():
        for name, parameter in experts.named_parameters():
            if name.endswith("lora_B"):
                parameter.normal_(0, 0.03)
    reference = copy.deepcopy(experts)
    experts[0].config.model.moe.blockwise_backend = backend
    experts[0].config.model.moe.grouped_token_budget = 3
    x = torch.randn(7, 16, dtype=torch.float64, requires_grad=True)
    other = x.detach().clone().requires_grad_()
    indices = torch.tensor([[0, 2], [1, 0], [0, 1], [2, 0], [2, 1], [1, 0], [0, 2]])
    weights = torch.rand(7, 2, dtype=torch.float64, requires_grad=True)
    other_weights = weights.detach().clone().requires_grad_()
    actual = grouped_experts(x, indices, weights, experts)
    expected = torch.zeros_like(other)
    for i, expert in enumerate(reference):
        tokens, slots = (indices == i).nonzero(as_tuple=True)
        if tokens.numel():
            expected = expected.index_add(
                0, tokens, expert(other[tokens]) * other_weights[tokens, slots, None]
            )
    coefficients = torch.randn_like(actual)
    (actual * coefficients).sum().backward()
    (expected * coefficients).sum().backward()
    torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-10)
    torch.testing.assert_close(x.grad, other.grad, atol=1e-12, rtol=1e-10)
    torch.testing.assert_close(weights.grad, other_weights.grad, atol=1e-12, rtol=1e-10)
    for (name, p), (other_name, q) in zip(
        experts.named_parameters(), reference.named_parameters(), strict=True
    ):
        assert name == other_name
        if not p.requires_grad or name.startswith("3."):
            assert p.grad is None and q.grad is None, name
        else:
            assert p.grad is not None and q.grad is not None, name
            torch.testing.assert_close(p.grad, q.grad, atol=1e-12, rtol=1e-10, msg=name)


@pytest.mark.parametrize("dropout", [0.0, 0.2])
def test_scheduled_lora_replays_dropout_and_adapter_roundtrip(monkeypatch, tmp_path, dropout):
    torch.manual_seed(19)
    native, _, config = gemma4_pair(
        monkeypatch,
        "A4B",
        lora=True,
        lora_targets=["gate_proj", "up_proj", "down_proj"],
        lora_dropout=dropout,
    )
    config.model.moe.grouped_token_budget = 3
    with torch.no_grad():
        for name, p in native.named_parameters():
            if name.endswith("lora_B"):
                p.normal_(0, 0.01)
    reference = copy.deepcopy(native)
    tokens = torch.tensor([[2, 3, 4, 5, 6, 7]])
    for model, backend in ((reference, "torch"), (native, "scheduled")):
        model.config.model.moe.expert_backend = "grouped"
        model.config.model.moe.blockwise_backend = backend
        reset_tensor_parallel_rng_tracker()
        result, _ = model(tokens)
        result.square().mean().backward()
    for (name, p), (_, q) in zip(
        native.named_parameters(), reference.named_parameters(), strict=True
    ):
        if p.grad is None:
            assert q.grad is None, name
        else:
            torch.testing.assert_close(p.grad, q.grad, atol=1e-6, rtol=1e-4, msg=name)
    native.eval()
    expected, _ = native(tokens)
    save_lora_adapter(native, tmp_path)
    with torch.no_grad():
        for name, p in native.named_parameters():
            if "lora_" in name:
                p.zero_()
    load_lora_adapter(native, tmp_path)
    restored, _ = native(tokens)
    torch.testing.assert_close(restored, expected, atol=0, rtol=0)
    merge_lora_weights(native)
    # Expert adapters are separate from base modules, so merging must fold
    # them into fused gate/up and down weights as well.
    assert not any("lora_" in n for n, _ in native.named_parameters())
    merged, _ = native(tokens)
    torch.testing.assert_close(merged, expected, atol=2e-5, rtol=2e-5)
