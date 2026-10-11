# Copyright (c) 2025-2026 Jaegeun Han
# SPDX-License-Identifier: Apache-2.0
"""Named SwiGLU LoRA targets must modify the matching mathematical branch."""

import pytest
import torch
import torch.nn.functional as F
from tests.fixtures.lora_tp import smollm2_lora_model

from ironcore.layers.mlp import MLP


@pytest.mark.parametrize("target", ["gate_proj", "up_proj"])
@pytest.mark.parametrize("activation", ["swiglu", "glu"])
def test_single_lora_target_matches_independent_projection(monkeypatch, target, activation):
    base = smollm2_lora_model(monkeypatch, lora=False)
    config = base.config
    config.peft.method = "lora"
    config.peft.lora.target_modules = [target]
    config.model.activation_type = activation
    native = MLP(config).double().eval()
    generator = torch.Generator().manual_seed(3407)
    x = torch.randn(2, 3, 32, generator=generator, dtype=torch.float64)
    gate = torch.randn(64, 32, generator=generator, dtype=torch.float64) * 0.1
    up = torch.randn(64, 32, generator=generator, dtype=torch.float64) * 0.1
    down = torch.randn(32, 64, generator=generator, dtype=torch.float64) * 0.1
    adapter = native.up_proj.lora_adapters[0]
    with torch.no_grad():
        branches = [gate.T, up.T] if activation == "swiglu" else [up.T, gate.T]
        native.up_proj.base_layer.weight.copy_(torch.cat(branches, dim=1))
        native.down_proj.weight.copy_(down.T)
        adapter.lora_A.normal_(0, 0.1, generator=generator)
        adapter.lora_B.normal_(0, 0.1, generator=generator)
    delta = adapter.scaling * F.linear(F.linear(x, adapter.lora_A.T), adapter.lora_B.T)
    gate_output, up_output = F.linear(x, gate), F.linear(x, up)
    if target == "gate_proj":
        gate_output = gate_output + delta
    else:
        up_output = up_output + delta
    hidden = (
        F.silu(gate_output) * up_output
        if activation == "swiglu"
        else up_output * torch.sigmoid(gate_output)
    )
    expected = F.linear(hidden, down)
    torch.testing.assert_close(native(x), expected, atol=1e-10, rtol=1e-10)
