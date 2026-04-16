# Copyright (c) 2025-2026 Jaegeun Han
#
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass, field

from .config import BaseConfig
from .config_moe import MoEConfig


@dataclass
class BiasConfig(BaseConfig):
    """Fine-grained bias configuration for transformer sub-layers.

    Each field controls whether that specific projection has a learnable bias.
    For fused projections (KV, gate+up), both must have bias=True for the fused
    layer to have bias. If one has bias and the other doesn't, use zero mask.

    Defaults match GPT-2 (all True). For LLaMA-style models like Qwen2.5,
    typically only q, k, v are True.
    """

    q: bool = field(default=True, metadata={"help": "Query projection bias"})
    k: bool = field(default=True, metadata={"help": "Key projection bias"})
    v: bool = field(default=True, metadata={"help": "Value projection bias"})
    o: bool = field(default=True, metadata={"help": "Output projection bias"})
    gate: bool = field(default=True, metadata={"help": "MLP gate projection bias (SwiGLU)"})
    up: bool = field(default=True, metadata={"help": "MLP up projection bias"})
    down: bool = field(default=True, metadata={"help": "MLP down projection bias"})

    @classmethod
    def all_true(cls) -> "BiasConfig":
        """Create config with all biases enabled (GPT-2 style)."""
        return cls(q=True, k=True, v=True, o=True, gate=True, up=True, down=True)

    @classmethod
    def qkv_only(cls) -> "BiasConfig":
        """Create config with only Q/K/V biases (LLaMA/Qwen style)."""
        return cls(q=True, k=True, v=True, o=False, gate=False, up=False, down=False)

    @classmethod
    def none(cls) -> "BiasConfig":
        """Create config with no biases."""
        return cls(q=False, k=False, v=False, o=False, gate=False, up=False, down=False)

    @property
    def qkv(self) -> bool:
        """Check if Q, K, and V all have bias (for fused QKV layers)."""
        return self.q and self.k and self.v

    @property
    def kv(self) -> bool:
        """Check if both K and V have bias (for fused KV layers)."""
        return self.k and self.v

    @property
    def gate_up(self) -> bool:
        """Check if both gate and up have bias (for fused gate+up layers)."""
        return self.gate and self.up


@dataclass
class PositionalEmbeddingConfig(BaseConfig):
    """positional embedding options"""

    type: str = field(default="absolute", metadata={"help": "absolute positional embedding"})
    base: int = field(
        default=10_000, metadata={"help": "Rotary positional embedding's base factor"}
    )
    scaling_factor: float = field(
        default=1.0,
        metadata={"help": "Rotary scaling factor for the rotary embeddings."},
    )
    offset: int = field(
        default=0,
        metadata={"help": "Rotary positional embedding starting position in a sequence"},
    )


@dataclass
class KVCacheConfig(BaseConfig):
    """KV Cache configuration"""

    enabled: bool = field(default=True, metadata={"help": "Enable KV cache"})
    max_batch_size: int = field(default=32, metadata={"help": "Maximum batch size for cache"})
    max_seq_length: int = field(
        default=2048, metadata={"help": "Maximum sequence length for cache"}
    )


@dataclass
class ModelConfig(BaseConfig):
    """model configuration options"""

    name: str = field(default="gpt2", metadata={"help": "model name (e.g., gpt2, llama)"})

    d_model: int = field(default=512, metadata={"help": "model hidden dimension size"})
    d_ffn: int = field(default=2048, metadata={"help": "model feed forward dimension size"})
    num_layers: int = field(default=2, metadata={"help": "number of layers"})
    max_seq_len: int = field(default=512, metadata={"help": "max sequence length"})
    max_position_embeddings: int | None = field(
        default=None, metadata={"help": "max position embeddings (defaults to max_seq_len)"}
    )
    dropout_embd: float = field(default=0.1, metadata={"help": "dropout ratio in embedding"})
    dropout_attn: float = field(default=0.1, metadata={"help": "dropout ratio in attention"})
    dropout_mlp: float = field(default=0.1, metadata={"help": "dropout ratio in mlp"})
    attention_head_size: int = field(default=64, metadata={"help": "attention head size"})
    precision: str = field(default="bfloat16", metadata={"help": "model dtype"})

    reset_position_ids: bool = field(
        default=True,
        metadata={"help": "Reset position ids after end-of-document token."},
    )
    reset_attention_mask: bool = field(
        default=True,
        metadata={"help": "Reset self attention mask after end-of-document token."},
    )
    eod_mask_loss: bool = field(
        default=False, metadata={"help": "Mask loss for the end of document tokens."}
    )

    positional_embedding: PositionalEmbeddingConfig = field(
        default_factory=PositionalEmbeddingConfig
    )

    kv_cache: KVCacheConfig = field(default_factory=KVCacheConfig)

    add_pooler: bool = field(default=True, metadata={"help": "add pooler"})
    untie_embed: bool = field(default=False, metadata={"help": "untie lm head"})

    # Bias configuration: fine-grained control over which layers have bias
    bias: BiasConfig = field(
        default_factory=BiasConfig.all_true,
        metadata={"help": "Bias configuration for each projection type"},
    )
    layernorm_bias: bool = field(default=True, metadata={"help": "bias in layernorm layers"})

    fp32_residual_connection: bool = field(
        default=False, metadata={"help": "fp32 residual connection"}
    )
    fp16_lm_cross_entropy: bool = field(
        default=False, metadata={"help": "use fp16 precision in cross entropy"}
    )
    ln_type: str = field(default="layernorm", metadata={"help": "layernorm type"})
    ln_eps: float = field(default=1e-5, metadata={"help": "layernorm epsilon"})
    post_ln: bool = field(default=False, metadata={"help": "use post layer norm"})

    # attention attributes
    num_attention_heads: int = field(default=8, metadata={"help": "number of attention heads"})
    head_dim: int = field(default=128, metadata={"help": "attention head dimension"})
    activation_type: str = field(default="gelu", metadata={"help": "activation function type"})
    seq_len_q: int | None = field(default=None, metadata={"help": "query sequence length"})
    seq_len_kv: int | None = field(default=None, metadata={"help": "key/value sequence length"})
    num_attention_groups: int = field(
        default=1,
        metadata={"help": "number of key-value groups in grouped query attention"},
    )
    attention_dropout: float = field(default=0.1, metadata={"help": "dropout ratio in attention"})

    # HuggingFace compatibility
    hf_model_type: str | None = field(
        default=None, metadata={"help": "HuggingFace model_type for checkpoint compatibility"}
    )
    hf_architecture: str | None = field(
        default=None, metadata={"help": "HuggingFace architecture name (e.g., 'LlamaForCausalLM')"}
    )

    # tokenizer
    tokenizer_type: str = field(
        default="gpt2",
        metadata={"help": "tokenizer type like bbpe, spe or model: gpt2, llama"},
    )
    vocab_name_or_path: str = field(default="gpt2", metadata={"help": "vocab name or path"})
    merge_file_path: str | None = field(default=None, metadata={"help": "merge file path"})

    # Mixture of Experts
    moe: MoEConfig = field(default_factory=MoEConfig)

    def __post_init__(self):
        if self.ln_type not in ["layernorm", "rmsnorm"]:
            raise ValueError(f"Invalid layer norm type: {self.ln_type}")

        # Default max_position_embeddings to max_seq_len if not set
        if self.max_position_embeddings is None:
            self.max_position_embeddings = self.max_seq_len
        assert all(
            getattr(self, k) > 0
            for k in [
                "d_model",
                "d_ffn",
                "num_layers",
                "max_seq_len",
                "attention_head_size",
                "num_attention_heads",
                "head_dim",
            ]
        ), "Configs must be positive"
        assert all(
            0 <= getattr(self, k) <= 1
            for k in ["dropout_embd", "dropout_attn", "dropout_mlp", "attention_dropout"]
        ), "Dropouts must be [0, 1]"
