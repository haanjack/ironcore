# Copyright (c) 2025-2026 Jaegeun Han
# SPDX-License-Identifier: Apache-2.0
"""Granite MoE decoder configuration (distinct from the Llama architecture)."""

import math
from dataclasses import dataclass

from .config import BaseConfig


@dataclass
class GraniteMoeConfig(BaseConfig):
    embedding_multiplier: float = 1.0
    attention_multiplier: float = 1.0
    residual_multiplier: float = 1.0
    logits_scaling: float = 1.0

    def __post_init__(self):
        for name in (
            "embedding_multiplier",
            "attention_multiplier",
            "residual_multiplier",
            "logits_scaling",
        ):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"granitemoe.{name} must be finite and positive")


def model_config_from_granitemoe(hf_config: dict):
    from .config_model import BiasConfig, KVCacheConfig, ModelConfig, PositionalEmbeddingConfig
    from .config_moe import MoEConfig

    if hf_config.get("model_type") != "granitemoe":
        raise ValueError("Expected a granitemoe checkpoint")
    rope = hf_config.get("rope_parameters") or hf_config.get("rope_scaling") or {}
    if rope.get("rope_type", rope.get("type", "default")) != "default":
        raise ValueError("Granite MoE currently supports default RoPE only")
    if hf_config.get("attention_bias", False) or hf_config.get("hidden_act", "silu") != "silu":
        raise ValueError("Granite MoE requires bias-free attention and SiLU experts")
    heads = hf_config["num_attention_heads"]
    head_dim = hf_config["hidden_size"] // heads
    return ModelConfig(
        name="granitemoe",
        hf_model_type="granitemoe",
        hf_architecture="GraniteMoeForCausalLM",
        d_model=hf_config["hidden_size"],
        d_ffn=hf_config["intermediate_size"],
        num_layers=hf_config["num_hidden_layers"],
        num_attention_heads=heads,
        num_attention_groups=hf_config.get("num_key_value_heads", heads),
        head_dim=head_dim,
        max_seq_len=hf_config["max_position_embeddings"],
        max_position_embeddings=hf_config["max_position_embeddings"],
        dropout_embd=0.0,
        dropout_attn=hf_config.get("attention_dropout", 0.0),
        dropout_mlp=0.0,
        precision=hf_config.get("dtype") or hf_config.get("torch_dtype") or "bfloat16",
        activation_type="swiglu",
        ln_type="rmsnorm",
        ln_eps=hf_config.get("rms_norm_eps", 1e-6),
        bias=BiasConfig.none(),
        layernorm_bias=False,
        untie_embed=not hf_config.get("tie_word_embeddings", False),
        positional_embedding=PositionalEmbeddingConfig(
            type="rope", base=rope.get("rope_theta", hf_config.get("rope_theta", 10000.0))
        ),
        kv_cache=KVCacheConfig(enabled=False),
        reset_position_ids=False,
        reset_attention_mask=False,
        tokenizer_type="sentencepiece",
        vocab_name_or_path=hf_config.get("_name_or_path", ""),
        moe=MoEConfig(
            use_moe=True,
            num_shared_experts=0,
            num_routed_experts=hf_config["num_local_experts"],
            num_experts_per_token=hf_config["num_experts_per_tok"],
            expert_intermediate_size=hf_config["intermediate_size"],
            aux_loss_alpha=0.0,
            expert_backend="grouped",
            # HF grouped_mm rounds each weighted contribution to model dtype,
            # then reduces top-k in FP32 internally, writing model dtype once.
            expert_accumulation_precision="float32",
        ),
        granitemoe=GraniteMoeConfig(
            embedding_multiplier=hf_config.get("embedding_multiplier", 1.0),
            attention_multiplier=hf_config.get("attention_multiplier", head_dim**-0.5),
            residual_multiplier=hf_config.get("residual_multiplier", 1.0),
            logits_scaling=hf_config.get("logits_scaling", 1.0),
        ),
    )


def validate_granitemoe_runtime(config):
    if not config.model.is_granitemoe:
        return
    model, moe = config.model, config.model.moe
    model.granitemoe.__post_init__()
    if config.trainer.context_parallel_size != 1:
        raise ValueError("Granite MoE custom attention scaling is not yet supported with CP")
    if moe.expert_backend not in {"loop", "grouped"} or moe.blockwise_backend != "torch":
        raise ValueError("Granite MoE currently supports loop or grouped/torch expert execution")
    if (
        not moe.use_moe
        or moe.num_shared_experts != 0
        or moe.expert_model_parallel_size != 1
        or moe.router_bias
        or moe.router_jitter_noise
        or moe.aux_loss_alpha
        or moe.drop_tokens
        or moe.expert_capacity_factor is not None
    ):
        raise ValueError(
            "Granite MoE requires EP=1, no shared experts, auxiliary loss or token dropping"
        )
    if (
        model.activation_type != "swiglu"
        or model.ln_type != "rmsnorm"
        or model.post_ln
        or model.fp32_residual_connection
        or model.dropout_mlp
        or model.dropout_embd
        or any(vars(model.bias).values())
        or model.layernorm_bias
        or model.positional_embedding.type != "rope"
        or model.positional_embedding.scaling_factor != 1.0
        or model.positional_embedding.offset != 0
    ):
        raise ValueError("Granite MoE requires bias-free pre-RMSNorm, SwiGLU and RoPE")
    if model.d_model != model.head_dim * model.num_attention_heads:
        raise ValueError("Granite MoE hidden size must match its attention head layout")
    if (moe.expert_intermediate_size or model.d_ffn) != model.d_ffn:
        raise ValueError("Granite MoE d_ffn must equal its expert intermediate size")
    if config.peft.method == "lora" and config.peft.lora.dropout:
        raise ValueError("Granite parameter LoRA requires zero dropout, matching HF PEFT")
    if config.peft.method == "lora":
        supported = {"q_proj", "k_proj", "v_proj", "o_proj", "gate_up_proj", "down_proj"}
        if set(config.peft.lora.target_modules) - supported:
            raise ValueError(
                "Granite expert LoRA targets the fused gate_up_proj and down_proj parameters"
            )
