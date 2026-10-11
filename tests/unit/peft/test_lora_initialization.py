# Copyright (c) 2025-2026 Jaegeun Han
# SPDX-License-Identifier: Apache-2.0
"""Native transposed A must use the input dimension for Kaiming fan-in."""

import math

import torch

from ironcore.peft.lora import LoRALinear


def test_lora_a_initialization_uses_input_width_and_keeps_zero_output():
    torch.manual_seed(71)
    adapter = LoRALinear(1024, 128, rank=8, alpha=16)
    bound = 1 / math.sqrt(1024)
    assert float(adapter.lora_A.detach().abs().max()) <= bound
    expected_std = bound / math.sqrt(3)
    assert abs(float(adapter.lora_A.detach().std()) - expected_std) < expected_std * 0.05
    assert torch.count_nonzero(adapter.lora_B) == 0
    assert torch.count_nonzero(adapter(torch.randn(2, 1024))) == 0
    first = adapter.lora_A.detach().clone()
    generator = torch.Generator().manual_seed(37)
    adapter._init_weights(generator)
    expected = adapter.lora_A.detach().clone()
    assert not torch.equal(first, expected)
    adapter._init_weights(torch.Generator().manual_seed(37))
    assert torch.equal(adapter.lora_A, expected)
