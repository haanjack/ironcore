# Copyright (c) 2025-2026 Jaegeun Han
# SPDX-License-Identifier: Apache-2.0

"""Fixed-seed projection output and full backward references for LoRA."""

import pytest
import torch
from tests.fixtures.config_fixtures import create_test_config

from ironcore.parallel import parallel_states
from ironcore.parallel.tensor_parallel import ColumnParallelLinear, RowParallelLinear
from ironcore.peft.lora import (
    LoRAColumnParallelLinear,
    LoRAConcatenatedColumnParallel,
    LoRARowParallelLinear,
)


@pytest.mark.parametrize("kind", ["column", "fused", "fused_subset", "row", "row_async"])
def test_lora_projection_forward_and_backward_values(monkeypatch, kind):
    torch.manual_seed(42)
    monkeypatch.setattr(parallel_states, "_TENSOR_MODEL_PARALLEL_WORLD_SIZE", 1)
    config = create_test_config()
    config.peft.lora.r = 2
    config.peft.lora.alpha = 4
    config.peft.lora.target_modules = ["k_proj", "v_proj"] if kind == "fused" else ["v_proj"]
    if kind.startswith("row"):
        base = RowParallelLinear(config, 8, 12, bias=True, input_is_parallel=True)
        layer = LoRARowParallelLinear(base, config.peft.lora)
    elif kind.startswith("fused"):
        base = ColumnParallelLinear(config, 8, 12, bias=True, concatenated_weights=2)
        layer = LoRAConcatenatedColumnParallel(base, config.peft.lora, ["k_proj", "v_proj"])
    else:
        base = ColumnParallelLinear(config, 8, 12, bias=True)
        layer = LoRAColumnParallelLinear(base, config.peft.lora)
    for parameter in layer.parameters():
        with torch.no_grad():
            parameter.copy_(torch.randn_like(parameter) * 0.1)
    x = torch.randn(2, 3, 8, requires_grad=True)
    ref_x = x.detach().clone().requires_grad_(True)
    params = {name: p.detach().clone().requires_grad_(True) for name, p in layer.named_parameters()}
    expected = ref_x @ params["base_layer.weight"] + params["base_layer.bias"]
    if kind.startswith("fused"):
        delta = []
        for index in range(2):
            if index in layer.adapter_map:
                prefix = f"lora_adapters.{layer.adapter_map[index]}"
                delta.append(
                    2.0 * ((ref_x @ params[prefix + ".lora_A"]) @ params[prefix + ".lora_B"])
                )
            else:
                delta.append(torch.zeros(2, 3, 6))
        expected = expected + torch.cat(delta, dim=-1)
    else:
        expected = expected + 2.0 * ((ref_x @ params["lora.lora_A"]) @ params["lora.lora_B"])
    if kind == "row_async":
        output, handle = layer(x, async_communication=True)
        actual = layer.finalize(output, handle)
    else:
        actual = layer(x)
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)
    actual.square().sum().backward()
    expected.square().sum().backward()
    torch.testing.assert_close(x.grad, ref_x.grad, atol=1e-6, rtol=1e-6)
    for name, p in layer.named_parameters():
        torch.testing.assert_close(p.grad, params[name].grad, atol=1e-6, rtol=1e-6)


def test_language_model_restores_zero_lora_b_with_distinct_seeded_a(monkeypatch):
    from tests.fixtures.lora_tp import smollm2_lora_model

    torch.manual_seed(42)
    model = smollm2_lora_model(monkeypatch)
    adapters = {name: p for name, p in model.named_parameters() if "lora_" in name}
    assert adapters
    for name, p in adapters.items():
        assert not getattr(p, "is_tp_sharded", False)
        if name.endswith("lora_B"):
            assert p.count_nonzero() == 0
    assert not torch.equal(
        model.model.layers[0].linear_q.lora.lora_A, model.model.layers[1].linear_q.lora.lora_A
    )
    torch.manual_seed(42)
    again = smollm2_lora_model(monkeypatch)
    for name, p in again.named_parameters():
        if name in adapters:
            torch.testing.assert_close(p, adapters[name], atol=0, rtol=0)
