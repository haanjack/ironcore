# Copyright (c) 2025-2026 Jaegeun Han
# SPDX-License-Identifier: Apache-2.0
"""Standalone adapters exclude the base and reject incompatible loads atomically."""

import pytest
import torch
from safetensors.torch import load_file
from tests.fixtures.gemma4 import gemma4_pair

from ironcore.peft import load_lora_adapter, save_lora_adapter


def test_adapter_roundtrip_preserves_logits_and_excludes_base(monkeypatch, tmp_path):
    native, _, _ = gemma4_pair(monkeypatch, "A4B", lora=True)
    native.eval()
    with torch.no_grad():
        for name, parameter in native.named_parameters():
            if name.endswith("lora_B"):
                parameter.fill_(0.01)
        expected, _ = native(torch.tensor([[2, 3, 4, 5]]))
    save_lora_adapter(native, tmp_path)
    weights = load_file(str(tmp_path / "adapter_model.safetensors"))
    assert weights and all("lora_" in name for name in weights)
    with torch.no_grad():
        for parameter in native.parameters():
            if parameter.requires_grad:
                parameter.zero_()
    load_lora_adapter(native, tmp_path)
    with torch.no_grad():
        actual, _ = native(torch.tensor([[2, 3, 4, 5]]))
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


def test_incompatible_adapter_scaling_is_rejected(monkeypatch, tmp_path):
    native, _, config = gemma4_pair(monkeypatch, lora=True)
    save_lora_adapter(native, tmp_path)
    config.peft.lora.alpha += 1
    with pytest.raises(ValueError, match="configuration mismatch"):
        load_lora_adapter(native, tmp_path)
