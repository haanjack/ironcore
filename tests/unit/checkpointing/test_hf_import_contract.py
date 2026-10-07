# Copyright (c) 2025-2026 Jaegeun Han
# SPDX-License-Identifier: Apache-2.0
import copy

import pytest
import torch

from ironcore.checkpointing import hf_interop
from ironcore.parallel import parallel_states
from ironcore.peft.lora import LoRALinear


def test_hf_base_import_loads_frozen_lora_wrapper_and_preserves_fresh_adapters(
    monkeypatch, tmp_path
):
    class Wrapper(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.base_layer = torch.nn.Linear(5, 3, bias=False).requires_grad_(False)
            self.lora = LoRALinear(5, 3, rank=2, alpha=2)

    model = torch.nn.Module()
    model.proj = Wrapper()
    initial = copy.deepcopy(model.proj.lora.state_dict())
    expected = torch.arange(15, dtype=torch.float32).reshape(3, 5)

    class Mapper:
        def __init__(self, *args):
            pass

        def hf_to_ironcore(self, *args, **kwargs):
            return {"proj.weight": expected.clone()}

    monkeypatch.setattr(hf_interop, "WeightMapper", Mapper)
    monkeypatch.setattr(hf_interop, "load_hf_config", lambda _: {"num_hidden_layers": 1})
    monkeypatch.setattr(hf_interop, "load_hf_state_dict", lambda *args, **kwargs: {})
    parallel_states.initialize_model_parallel(1, 2)
    try:
        result = hf_interop.load_from_huggingface(tmp_path, model, architecture="llama")
        torch.testing.assert_close(model.proj.base_layer.weight, expected, atol=0, rtol=0)
        for name, value in model.proj.lora.state_dict().items():
            torch.testing.assert_close(value, initial[name], atol=0, rtol=0)
        hf_interop.validate_imported_base_parameters(model, result["missing_keys"])
        with pytest.raises(ValueError, match="base parameters"):
            hf_interop.validate_imported_base_parameters(model, ["proj.base_layer.weight"])
    finally:
        parallel_states.destroy_model_parallel()
