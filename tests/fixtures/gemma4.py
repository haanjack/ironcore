# Copyright (c) 2025-2026 Jaegeun Han
#
# SPDX-License-Identifier: Apache-2.0

"""Tiny dense and MoE Gemma 4 variants for download-free parity tests."""

from __future__ import annotations

import logging
from types import SimpleNamespace

from tests.fixtures.config_fixtures import create_test_config

from ironcore.config import MainConfig
from ironcore.config.config_gemma4 import model_config_from_gemma4
from ironcore.language_model import LanguageModel
from ironcore.parallel import parallel_states


def gemma4_pair(
    monkeypatch,
    variant: str = "E2B",
    lora: bool = False,
    tp_size: int = 1,
    lora_targets: list[str] | None = None,
    lora_dropout: float = 0.0,
    cp_size: int = 1,
) -> tuple[LanguageModel, object, MainConfig]:
    """Build IronCore and Transformers decoders with identical reference weights."""
    import pytest

    pytest.importorskip(
        "transformers.models.gemma4",
        reason="Gemma 4 numerical reference requires recent Transformers",
    )
    from transformers import Gemma4ForCausalLM, Gemma4TextConfig

    from ironcore.checkpointing.weight_mapping import Architecture, WeightMapper

    small = variant in {"E2B", "E4B"}
    hf_config = Gemma4TextConfig(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=1 if variant == "E2B" else 2,
        head_dim=8,
        global_head_dim=16,
        num_global_key_value_heads=None if small else 1,
        max_position_embeddings=32,
        sliding_window=3,
        layer_types=["sliding_attention", "full_attention", "sliding_attention", "full_attention"],
        hidden_size_per_layer_input=4 if small else 0,
        vocab_size_per_layer_input=32,
        num_kv_shared_layers=2 if small else 0,
        attention_k_eq_v=not small,
        use_double_wide_mlp=variant == "E2B",
        final_logit_softcapping=30.0,
        attention_dropout=0.0,
        dtype="float32",
        enable_moe_block=variant == "A4B",
        num_experts=4,
        top_k_experts=2,
        moe_intermediate_size=8,
    )
    hf_config._attn_implementation = "eager"
    reference = Gemma4ForCausalLM(hf_config).float()
    config = create_test_config(precision="float32", use_flash_attn=False)
    config.model = model_config_from_gemma4(hf_config.to_dict())
    config.model.precision = "float32"
    config.trainer.tensor_model_parallel_size = tp_size
    config.trainer.context_parallel_size = cp_size
    config.parallel.world_size = tp_size * cp_size
    if cp_size > 1:
        config.trainer.context_parallel_backend = "sdpa"
        config.model.gemma4.attention_chunk_size = 2
        config.data.task_type = "sft"
        config.data.sft_packing = False
    config.data.vocab_size = 32
    config.operation.activation_recompute = False
    if lora:
        config.peft.method = "lora"
        config.peft.lora.r = 2
        config.peft.lora.dropout = lora_dropout
        config.peft.lora.target_modules = lora_targets or [
            "q_proj",
            "v_proj",
            "gate_proj",
            "down_proj",
        ]
    config.model.reset_position_ids = False
    config.model.reset_attention_mask = False
    tokenizer = SimpleNamespace(vocab_size=32, padded_vocab_size=32, eod_token_id=1, pad_token_id=0)
    monkeypatch.setattr("ironcore.language_model.get_tokenizer", lambda: tokenizer)
    monkeypatch.setattr("ironcore.layers.embedding.get_tokenizer", lambda: tokenizer)
    if tp_size == 1:
        monkeypatch.setattr(parallel_states, "_TENSOR_MODEL_PARALLEL_WORLD_SIZE", 1)
        monkeypatch.setattr(parallel_states, "_DATA_PARALLEL_WORLD_SIZE", 1)
        monkeypatch.setattr(parallel_states, "get_tensor_model_parallel_rank", lambda: 0)
    from ironcore import global_vars
    from ironcore.utils import Timer

    monkeypatch.setattr(
        global_vars,
        "GLOBAL_STATES",
        SimpleNamespace(
            timer=Timer(),
            get_logger=lambda: logging.getLogger("gemma4-tests"),
            get_tokenizer=lambda: tokenizer,
        ),
    )
    native = LanguageModel(config).float()
    if tp_size > 1:
        import json
        import tempfile
        from pathlib import Path

        from safetensors.torch import save_file

        from ironcore.checkpointing.hf_interop import load_from_huggingface

        # Exercise the production HF loader's TP splitting, including PLE.
        with tempfile.TemporaryDirectory() as checkpoint:
            root = Path(checkpoint)
            (root / "config.json").write_text(json.dumps(hf_config.to_dict()))
            save_file(
                {
                    key: value
                    for key, value in reference.state_dict().items()
                    if key != "lm_head.weight"
                },
                root / "model.safetensors",
            )
            info = load_from_huggingface(checkpoint, native, strict=not lora)
            assert not info["unexpected_keys"]
            assert all(".lora." in name for name in info["missing_keys"])
        if lora:
            from ironcore.peft.utils import freeze_base_model

            freeze_base_model(native, "lora")
        return native, reference, config
    mapped = WeightMapper(Architecture.GEMMA4, 4).hf_to_ironcore(reference.state_dict())
    if lora:
        from ironcore.peft.utils import freeze_base_model

        state = {}
        for name in native.state_dict():
            canonical = name.replace(".base_layer.", ".")
            if canonical in mapped:
                state[name] = mapped[canonical]
        missing, unexpected = native.load_state_dict(state, strict=False)
        assert not unexpected
        assert all(".lora." in name for name in missing)
        freeze_base_model(native, "lora")
    else:
        native.load_state_dict(mapped, strict=True)
    return native, reference, config
