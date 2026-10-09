# Copyright (c) 2025-2026 Jaegeun Han
#
# SPDX-License-Identifier: Apache-2.0

"""Architecture-specific options for Gemma 4 dense and MoE text decoders."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from .config import BaseConfig

if TYPE_CHECKING:
    from . import MainConfig
    from .config_model import ModelConfig


def validate_gemma4_runtime(config: MainConfig) -> None:
    """Require supported causal decoder layouts and execution options."""
    model = config.model
    if not model.is_gemma4:
        return
    model.gemma4.validate(model)
    tp_size = config.trainer.tensor_model_parallel_size
    if tp_size < 1 or model.num_attention_heads % tp_size or model.d_ffn % tp_size:
        raise ValueError("Gemma 4 query heads and MLP width must be divisible by TP size")
    gemma = model.gemma4
    if gemma.hidden_size_per_layer_input and (
        gemma.hidden_size_per_layer_input % tp_size or gemma.vocab_size_per_layer_input % tp_size
    ):
        raise ValueError("Gemma 4 PLE width and vocabulary must be divisible by TP size")
    for layer_idx in range(model.num_layers):
        head_dim, kv_heads = gemma.head_layout(model, layer_idx)
        if kv_heads == 1 and tp_size > 1:
            if head_dim % tp_size:
                raise ValueError(
                    "Gemma 4 replicated KV projection width must be divisible by TP size"
                )
        elif kv_heads % tp_size:
            raise ValueError(
                "Gemma 4 KV heads must be divisible by TP size, or a single replicated head"
            )
    if model.moe.use_moe:
        moe = model.moe
        if moe.expert_backend not in {"loop", "grouped"}:
            raise ValueError("Gemma 4 MoE supports loop or grouped experts")
        if (
            moe.num_shared_experts != 1
            or moe.aux_loss_alpha
            or moe.router_jitter_noise
            or moe.router_bias
            or moe.drop_tokens
            or moe.expert_capacity_factor is not None
        ):
            raise ValueError("Gemma 4 MoE requires its native router and one shared MLP")
        if model.moe.expert_model_parallel_size != 1:
            raise ValueError("Gemma 4 MoE currently requires EP=1")
        if not model.moe.expert_intermediate_size or model.moe.expert_intermediate_size % tp_size:
            raise ValueError("Gemma 4 expert width must be divisible by TP size")
    if model.untie_embed or any(
        getattr(model.bias, name) for name in ("q", "k", "v", "o", "gate", "up", "down")
    ):
        raise ValueError("Gemma 4 requires tied embeddings and bias-free projections")
    if model.positional_embedding.type != "rope" or model.activation_type != "gelu_pytorch_tanh":
        raise ValueError("Gemma 4 requires RoPE and gelu_pytorch_tanh activation")
    if model.ln_type != "rmsnorm" or model.layernorm_bias:
        raise ValueError("Gemma 4 requires bias-free RMSNorm")
    if model.dropout_embd or model.dropout_mlp:
        raise ValueError("Gemma 4 does not use embedding or MLP dropout")
    if model.kv_cache.use_paged:
        raise ValueError("Gemma 4 uses explicit KV tuples; paged KV cache is not supported")
    if config.offload.activation_spill or config.offload.weight_offload:
        if gemma.hidden_size_per_layer_input or gemma.num_kv_shared_layers:
            raise ValueError("Gemma 4 offload currently requires no PLE or shared KV")
        if config.offload.activation_spill_granularity != "full_layer":
            raise ValueError("Gemma 4 offload requires full_layer activation spilling")


@dataclass
class Gemma4Config(BaseConfig):
    """Keep local/global attention and per-layer input settings together."""

    layer_types: list[str] = field(default_factory=list)
    sliding_window: int = 512
    global_head_dim: int = 512
    num_global_key_value_heads: int | None = None
    sliding_rope_theta: float = 10_000.0
    global_rope_theta: float = 1_000_000.0
    global_partial_rotary_factor: float = 0.25
    hidden_size_per_layer_input: int = 256
    vocab_size_per_layer_input: int = 262_144
    num_kv_shared_layers: int = 0
    attention_k_eq_v: bool = False
    use_double_wide_mlp: bool = False
    final_logit_softcapping: float | None = 30.0
    pad_token_id: int = 0
    attention_chunk_size: int | None = None

    def validate(self, model: ModelConfig) -> None:
        """Reject inconsistent layouts before allocating decoder weights."""
        if not self.layer_types:
            self.layer_types = [
                "full_attention" if (i + 1) % 6 == 0 else "sliding_attention"
                for i in range(model.num_layers)
            ]
            self.layer_types[-1] = "full_attention"
        if len(self.layer_types) != model.num_layers or any(
            kind not in {"sliding_attention", "full_attention"} for kind in self.layer_types
        ):
            raise ValueError("gemma4.layer_types must specify one supported type per layer")
        if self.layer_types[-1] != "full_attention":
            raise ValueError("Gemma 4's final layer must use full_attention")
        if self.sliding_window < 1 or self.global_head_dim < 2 or self.global_head_dim % 2:
            raise ValueError("Gemma 4 needs a positive sliding window and even global head_dim")
        if (
            model.num_attention_groups < 1
            or model.head_dim % 2
            or model.num_attention_heads % model.num_attention_groups
        ):
            raise ValueError("Gemma 4 needs even head_dim and query heads divisible by KV heads")
        if self.num_global_key_value_heads is not None and (
            self.num_global_key_value_heads < 1
            or model.num_attention_heads % self.num_global_key_value_heads
        ):
            raise ValueError("Gemma 4 query heads must be divisible by global KV heads")
        if not 0 < self.global_partial_rotary_factor <= 1:
            raise ValueError("gemma4.global_partial_rotary_factor must be in (0, 1]")
        if self.sliding_rope_theta <= 0 or self.global_rope_theta <= 0:
            raise ValueError("Gemma 4 RoPE bases must be positive")
        if not 0 <= self.num_kv_shared_layers < model.num_layers:
            raise ValueError("gemma4.num_kv_shared_layers must be smaller than num_layers")
        if self.hidden_size_per_layer_input < 0 or self.vocab_size_per_layer_input < 1:
            raise ValueError(
                "Gemma 4 PLE dimensions must be nonnegative with a positive vocabulary"
            )
        if not 0 <= self.pad_token_id < self.vocab_size_per_layer_input:
            raise ValueError("Gemma 4 pad_token_id must be in the PLE vocabulary")
        if self.final_logit_softcapping is not None and self.final_logit_softcapping <= 0:
            raise ValueError("gemma4.final_logit_softcapping must be positive or None")
        if self.attention_chunk_size is not None and (
            not isinstance(self.attention_chunk_size, int)
            or isinstance(self.attention_chunk_size, bool)
            or self.attention_chunk_size < 1
        ):
            raise ValueError("Gemma 4 attention_chunk_size must be a positive integer or None")
        if self.num_kv_shared_layers:
            start = model.num_layers - self.num_kv_shared_layers
            available = set(self.layer_types[:start])
            if not set(self.layer_types[start:]).issubset(available):
                raise ValueError("Shared Gemma 4 layers need an earlier KV producer of each type")

    def head_layout(self, model: ModelConfig, layer_idx: int) -> tuple[int, int]:
        """Return this layer's head dimension and number of KV heads."""
        if self.layer_types[layer_idx] == "full_attention":
            groups = (
                self.num_global_key_value_heads
                if self.attention_k_eq_v and self.num_global_key_value_heads is not None
                else model.num_attention_groups
            )
            return self.global_head_dim, groups
        return model.head_dim, model.num_attention_groups


def model_config_from_gemma4(hf_config: dict) -> ModelConfig:
    """Translate a Gemma 4 text or multimodal HF config into a native text config."""
    from .config_model import BiasConfig, KVCacheConfig, ModelConfig, PositionalEmbeddingConfig
    from .config_moe import MoEConfig

    text = hf_config.get("text_config", hf_config)
    if text.get("model_type") != "gemma4_text":
        raise ValueError("Only Gemma 4 text configurations are supported")
    rope = text.get("rope_parameters", {})
    local = rope.get("sliding_attention", {})
    global_rope = rope.get("full_attention", {})
    if (
        local.get("rope_type", "default") != "default"
        or global_rope.get("rope_type", "proportional") != "proportional"
    ):
        raise ValueError("Gemma 4 requires default local RoPE and proportional global RoPE")
    if local.get("factor", 1.0) != 1.0 or global_rope.get("factor", 1.0) != 1.0:
        raise ValueError("Additional Gemma 4 RoPE scaling factors are not supported")
    if text.get("use_bidirectional_attention") not in {None, "vision"}:
        raise ValueError("Gemma 4 text training requires causal attention")
    if text.get("attention_bias", False) or not text.get("tie_word_embeddings", True):
        raise ValueError("Gemma 4 dense support requires bias-free projections and tied embeddings")
    if text.get("hidden_activation", "gelu_pytorch_tanh") != "gelu_pytorch_tanh":
        raise ValueError("Gemma 4 dense support requires gelu_pytorch_tanh")
    # Recent Transformers versions serialize legacy global_head_dim as indexed
    # per_layer_config overrides. Accept both layouts and reject heterogeneous
    # overrides that the native configuration cannot express.
    global_dim = text.get("global_head_dim", 512)
    global_groups = text.get("num_global_key_value_heads")
    overrides = text.get("per_layer_config", {})
    full_layers = [
        i for i, kind in enumerate(text.get("layer_types", [])) if kind == "full_attention"
    ]
    layouts = [overrides.get(str(i), overrides.get(i, {})) for i in full_layers]
    if layouts:
        dims = {layout.get("head_dim", global_dim) for layout in layouts}
        groups = {layout.get("num_key_value_heads", global_groups) for layout in layouts}
        if len(dims) != 1 or len(groups) != 1:
            raise ValueError("Gemma 4 full-attention layers must have a consistent head layout")
        global_dim = dims.pop()
        global_groups = groups.pop()
    return ModelConfig(
        name="gemma4",
        d_model=text["hidden_size"],
        d_ffn=text["intermediate_size"],
        num_layers=text["num_hidden_layers"],
        num_attention_heads=text["num_attention_heads"],
        num_attention_groups=text["num_key_value_heads"],
        head_dim=text["head_dim"],
        max_seq_len=text["max_position_embeddings"],
        max_position_embeddings=text["max_position_embeddings"],
        dropout_attn=text.get("attention_dropout", 0.0),
        dropout_mlp=0.0,
        dropout_embd=0.0,
        precision=text.get("dtype") or "bfloat16",
        ln_type="rmsnorm",
        ln_eps=text.get("rms_norm_eps", 1e-6),
        layernorm_bias=False,
        bias=BiasConfig.none(),
        activation_type="gelu_pytorch_tanh",
        positional_embedding=PositionalEmbeddingConfig(type="rope"),
        kv_cache=KVCacheConfig(enabled=True),
        reset_position_ids=False,
        reset_attention_mask=False,
        hf_model_type="gemma4_text",
        hf_architecture="Gemma4ForCausalLM",
        tokenizer_type="sentencepiece",
        vocab_name_or_path=hf_config.get("_name_or_path", "google/gemma-4-E2B-it"),
        moe=MoEConfig(
            use_moe=text.get("enable_moe_block", False),
            num_shared_experts=1,
            num_routed_experts=text.get("num_experts") or 128,
            num_experts_per_token=text.get("top_k_experts") or 8,
            expert_intermediate_size=text.get("moe_intermediate_size") or 704,
            aux_loss_alpha=0.0,
        ),
        gemma4=Gemma4Config(
            layer_types=text.get("layer_types", []),
            sliding_window=text.get("sliding_window", 512),
            global_head_dim=global_dim,
            num_global_key_value_heads=global_groups,
            sliding_rope_theta=local.get("rope_theta", 10_000.0),
            global_rope_theta=global_rope.get("rope_theta", 1_000_000.0),
            global_partial_rotary_factor=global_rope.get("partial_rotary_factor", 0.25),
            hidden_size_per_layer_input=text.get("hidden_size_per_layer_input", 256),
            vocab_size_per_layer_input=text.get("vocab_size_per_layer_input", text["vocab_size"]),
            num_kv_shared_layers=text.get("num_kv_shared_layers", 0),
            attention_k_eq_v=text.get("attention_k_eq_v", False),
            use_double_wide_mlp=text.get("use_double_wide_mlp", False),
            final_logit_softcapping=text.get("final_logit_softcapping"),
            pad_token_id=text.get("pad_token_id", 0),
        ),
    )
