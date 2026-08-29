# Copyright (c) 2025-2026 Jaegeun Han
#
# SPDX-License-Identifier: Apache-2.0

"""
Regression test: export_to_huggingface() must use the authoritative training
config (via HFConfigManager.get_hf_config) when available, instead of always
guessing config values from the model's tensor shapes.

Root cause: export_to_huggingface() previously had no way to see the
MainConfig the model was trained with, so it always fell back to guessing
(e.g. intermediate_size = 4 * hidden_size regardless of the model's real FFN
width), while save_checkpoint()'s HF config used the real config values —
the two paths disagreed for any model with a non-4x FFN ratio.
"""

import torch

from ironcore.checkpointing.hf_interop import export_to_huggingface
from ironcore.checkpointing.native import HFConfigManager
from ironcore.config import (
    DataConfig,
    InitConfig,
    MainConfig,
    ModelConfig,
    OperationConfig,
    OptimConfig,
    ParallelConfig,
    PEFTConfig,
    ProfilerConfig,
    TrainerConfig,
    UtilsConfig,
)
from ironcore.parallel import parallel_states


class _TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = torch.nn.Module()
        self.model.layers = torch.nn.ModuleList([torch.nn.Linear(4, 4) for _ in range(2)])


def _make_config(tmp_path) -> MainConfig:
    return MainConfig(
        # Deliberately NOT a 4x ratio, to distinguish from the guess-based path.
        model=ModelConfig(
            d_model=8,
            d_ffn=20,
            num_layers=2,
            num_attention_heads=2,
            hf_model_type="llama",
            hf_architecture="LlamaForCausalLM",
        ),
        init=InitConfig(),
        optim=OptimConfig(),
        data=DataConfig(vocab_size=111),
        parallel=ParallelConfig(),
        trainer=TrainerConfig(model_path=str(tmp_path)),
        operation=OperationConfig(),
        utils=UtilsConfig(),
        profiler=ProfilerConfig(),
        peft=PEFTConfig(),
    )


def test_export_uses_authoritative_config_when_provided(tmp_path, monkeypatch):
    monkeypatch.setattr(parallel_states, "get_data_parallel_group_rank", lambda: 0)
    monkeypatch.setattr(parallel_states, "get_tensor_model_parallel_world_size", lambda: 1)
    monkeypatch.setattr(parallel_states, "get_tensor_model_parallel_rank", lambda: 0)

    config = _make_config(tmp_path)
    model = _TinyModel()

    result = export_to_huggingface(
        model,
        tmp_path / "export",
        architecture="llama",
        use_safetensors=False,
        ironcore_config=config,
    )

    import json

    with open(result["config_file"]) as f:
        exported = json.load(f)

    expected = HFConfigManager.get_hf_config(config)

    # The exported config must match the authoritative generator exactly,
    # not the shape-guessing fallback (which would compute
    # intermediate_size = 4 * hidden_size = 32, not the real d_ffn = 20).
    assert exported == expected
    assert exported["intermediate_size"] == 20
