# Copyright (c) 2025-2026 Jaegeun Han
# SPDX-License-Identifier: Apache-2.0

"""TP-compatible scaled SmolLM2 decoder for download-free LoRA comparisons."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from types import SimpleNamespace

import torch
from tests.fixtures.config_fixtures import create_test_config

from ironcore import global_vars
from ironcore.checkpointing.weight_mapping import Architecture, WeightMapper
from ironcore.config import ModelConfig
from ironcore.config.config_model import BiasConfig, PositionalEmbeddingConfig
from ironcore.language_model import LanguageModel
from ironcore.parallel import parallel_states
from ironcore.peft.utils import freeze_base_model
from ironcore.utils import Timer


def smollm2_lora_model(
    monkeypatch,
    tp_size: int = 1,
    checkpoint: Path | None = None,
    dropout: float = 0.0,
    *,
    cp_size: int = 1,
    cp_backend: str = "sdpa",
    lora: bool = True,
    moe: bool = False,
    mlp_chunk_size: int | None = None,
    expert_backend: str = "loop",
    blockwise_backend: str = "torch",
    ep_size: int = 1,
) -> LanguageModel:
    """Build a tiny Llama/SmolLM2 layout with all attention/MLP LoRA targets."""
    from transformers import LlamaConfig, LlamaForCausalLM

    hf_config = LlamaConfig(
        vocab_size=32,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=32,
        rms_norm_eps=1e-5,
        rope_theta=130000.0,
        tie_word_embeddings=True,
        attention_dropout=0.0,
    )
    if checkpoint:
        hf_config = LlamaConfig.from_dict(json.loads((checkpoint / "config.json").read_text()))
    reference = None if checkpoint else LlamaForCausalLM(hf_config).float()
    config = create_test_config(precision="float32", use_flash_attn=False)
    config.model = ModelConfig(
        name="smollm2",
        d_model=hf_config.hidden_size,
        d_ffn=hf_config.intermediate_size,
        num_layers=hf_config.num_hidden_layers,
        num_attention_heads=hf_config.num_attention_heads,
        num_attention_groups=hf_config.num_key_value_heads,
        head_dim=hf_config.hidden_size // hf_config.num_attention_heads,
        max_seq_len=32,
        max_position_embeddings=hf_config.max_position_embeddings,
        ln_type="rmsnorm",
        ln_eps=hf_config.rms_norm_eps,
        layernorm_bias=False,
        bias=BiasConfig.none(),
        activation_type="swiglu",
        positional_embedding=PositionalEmbeddingConfig(
            type="rope",
            base=hf_config.to_dict()
            .get("rope_parameters", {})
            .get("rope_theta", hf_config.to_dict().get("rope_theta", 10000.0)),
        ),
        dropout_attn=0.0,
        dropout_mlp=0.0,
        dropout_embd=0.0,
        precision="float32",
        untie_embed=False,
        reset_attention_mask=False,
        reset_position_ids=False,
    )
    if cp_size > 1 and cp_backend == "ring":
        config.model.precision = "bfloat16"
    config.trainer.tensor_model_parallel_size = tp_size
    config.trainer.mlp_chunk_size = mlp_chunk_size
    if moe:
        from ironcore.config.config_moe import MoEConfig

        config.model.moe = MoEConfig(
            use_moe=True,
            num_shared_experts=1,
            num_routed_experts=4,
            num_experts_per_token=2,
            aux_loss_alpha=0.03,
            expert_backend=expert_backend,
            blockwise_backend=blockwise_backend,
            expert_model_parallel_size=ep_size,
            virtual_block_size=3 if expert_backend == "grouped" else 128,
            grouped_token_budget=7 if expert_backend == "grouped" else 4096,
        )
    config.trainer.context_parallel_size = cp_size
    config.trainer.context_parallel_backend = cp_backend
    config.parallel.world_size = max(
        tp_size * cp_size,
        torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1,
    )
    config.operation.activation_recompute = False
    config.data.vocab_size = hf_config.vocab_size
    config.peft.method = "lora" if lora else "none"
    config.peft.lora.r = 2
    config.peft.lora.alpha = 4.0
    config.peft.lora.dropout = dropout
    config.peft.lora.target_modules = [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "up_proj",
        "gate_proj",
        "down_proj",
    ]
    tokenizer = SimpleNamespace(
        vocab_size=hf_config.vocab_size,
        padded_vocab_size=hf_config.vocab_size,
        eod_token_id=0,
        pad_token_id=0,
    )
    monkeypatch.setattr("ironcore.language_model.get_tokenizer", lambda: tokenizer)
    monkeypatch.setattr("ironcore.layers.embedding.get_tokenizer", lambda: tokenizer)
    monkeypatch.setattr(
        global_vars,
        "GLOBAL_STATES",
        SimpleNamespace(
            timer=Timer(),
            get_logger=lambda: logging.getLogger("lora-tp"),
            get_tokenizer=lambda: tokenizer,
        ),
    )
    if cp_size == 1:
        monkeypatch.setattr(parallel_states, "_CONTEXT_PARALLEL_WORLD_SIZE", 1)
    if tp_size == 1:
        monkeypatch.setattr(parallel_states, "_TENSOR_MODEL_PARALLEL_WORLD_SIZE", 1)
        if cp_size == 1:
            monkeypatch.setattr(parallel_states, "_DATA_PARALLEL_WORLD_SIZE", 1)
        monkeypatch.setattr(parallel_states, "get_tensor_model_parallel_rank", lambda: 0)
    native = LanguageModel(config).float()
    if checkpoint:
        from ironcore.checkpointing.hf_interop import (
            load_from_huggingface,
            validate_imported_base_parameters,
        )

        info = load_from_huggingface(checkpoint, native, architecture="llama")
        validate_imported_base_parameters(native, info["missing_keys"])
    elif tp_size == 1 and not moe:
        mapped = WeightMapper(Architecture.LLAMA, 2).hf_to_ironcore(
            reference.state_dict(), strict=False
        )
        state = {
            name: mapped[name.replace(".base_layer.", ".")]
            for name in native.state_dict()
            if name.replace(".base_layer.", ".") in mapped
        }
        native.load_state_dict(state, strict=False)
    if lora:
        freeze_base_model(native, "lora")
    return native


def shard_like(model: LanguageModel, name: str, full: torch.Tensor) -> torch.Tensor:
    """Select the native shard using checkpoint attributes; replicas stay full."""
    from ironcore.parallel.tensor_parallel import comm

    target = model.state_dict()[name]
    if full.shape == target.shape:
        return full
    module = model.get_submodule(name.rsplit(".", 1)[0])
    return comm.split_to_model_parallel_workers(
        full,
        {
            "column_parallel": module.column_parallel,
            "row_parallel": module.row_parallel,
            "concatenated_weights": module.concatenated_weights,
        },
    )
