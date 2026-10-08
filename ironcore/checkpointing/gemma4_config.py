# Copyright (c) 2025-2026 Jaegeun Han
#
# SPDX-License-Identifier: Apache-2.0

"""HF config serialization for the Gemma 4 dense text backbone."""

from __future__ import annotations

from ironcore.config import MainConfig


def get_gemma4_hf_config(config: MainConfig) -> dict:
    """Export a text-only Gemma4ForCausalLM config, including all PLE/KV options."""
    model, gemma = config.model, config.model.gemma4
    return {
        "model_type": "gemma4_text",
        "architectures": ["Gemma4ForCausalLM"],
        "hidden_size": model.d_model,
        "intermediate_size": model.d_ffn,
        "num_hidden_layers": model.num_layers,
        "num_attention_heads": model.num_attention_heads,
        "num_key_value_heads": model.num_attention_groups,
        "head_dim": model.head_dim,
        "global_head_dim": gemma.global_head_dim,
        "num_global_key_value_heads": gemma.num_global_key_value_heads,
        "rms_norm_eps": model.ln_eps,
        "hidden_activation": "gelu_pytorch_tanh",
        "max_position_embeddings": model.max_position_embeddings,
        "vocab_size": config.data.vocab_size,
        "dtype": model.precision,
        "initializer_range": config.init.init_std,
        "attention_bias": False,
        "attention_dropout": model.dropout_attn,
        "tie_word_embeddings": True,
        "use_cache": True,
        "pad_token_id": gemma.pad_token_id,
        "bos_token_id": 2,
        "eos_token_id": 1,
        "use_bidirectional_attention": None,
        "enable_moe_block": False,
        "layer_types": list(gemma.layer_types),
        "sliding_window": gemma.sliding_window,
        "attention_k_eq_v": gemma.attention_k_eq_v,
        "num_kv_shared_layers": gemma.num_kv_shared_layers,
        "hidden_size_per_layer_input": gemma.hidden_size_per_layer_input,
        "vocab_size_per_layer_input": gemma.vocab_size_per_layer_input,
        "use_double_wide_mlp": gemma.use_double_wide_mlp,
        "final_logit_softcapping": gemma.final_logit_softcapping,
        "rope_parameters": {
            "sliding_attention": {"rope_type": "default", "rope_theta": gemma.sliding_rope_theta},
            "full_attention": {
                "rope_type": "proportional",
                "rope_theta": gemma.global_rope_theta,
                "partial_rotary_factor": gemma.global_partial_rotary_factor,
            },
        },
    }
