# Copyright (c) 2025-2026 Jaegeun Han
#
# SPDX-License-Identifier: MIT
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the above copyright notice,
# this list of conditions, and the following disclaimer are retained.
#
# Full license text is available at LICENSE file.

from dataclasses import dataclass, field
from typing import Optional

from .config import BaseConfig


@dataclass
class PositionalEmbeddingConfig(BaseConfig):
    """positional embedding options"""

    type: str = field(
        default="absolute", metadata={"help": "absolute positional embedding"}
    )
    base: int = field(
        default=10_000, metadata={"help": "Rotary positional embedding's base factor"}
    )
    scaling_factor: float = field(
        default=1.0,
        metadata={"help": "Rotary sscaling factor for the rotary embeddings."},
    )
    offset: int = field(
        default=0,
        metadata={
            "help": "Rotary positional embedding starting position in a sequence"
        },
    )


@dataclass
class ModelConfig(BaseConfig):
    """model configuration options"""

    d_model: int = field(default=512, metadata={
                         "help": "model hidden dimension size"})
    d_ffn: int = field(
        default=2048, metadata={"help": "model feed forward dimension size"}
    )
    num_layers: int = field(default=2, metadata={"help": "number of layers"})
    num_attention_heads: int = field(
        default=8, metadata={"help": "number of attention heads"}
    )
    max_len: int = field(default=512, metadata={"help": "max sequence length"})
    max_position_embeddings: int = field(
        default=512, metadata={"help": "max position embeddings"}
    )
    dropout_embd: float = field(
        default=0.1, metadata={"help": "dropout ratio in embedding"}
    )
    dropout_attn: float = field(
        default=0.1, metadata={"help": "dropout ratio in attention"}
    )
    dropout_mlp: float = field(default=0.1, metadata={
                               "help": "dropout ratio in mlp"})
    attention_head_size: int = field(
        default=64, metadata={"help": "attention head size"}
    )
    max_seq_len: int = field(default=512, metadata={
                             "help": "max sequence length"})
    precision: str = field(default="bfloat16", metadata={
                           "help": "model dtype"})

    reset_position_ids: bool = field(
        default=True,
        metadata={"help": "Reset posistion ids after end-of-document token."},
    )
    reset_attention_mask: bool = field(
        default=True,
        metadata={
            "help": "Reset self attention maske after " "end-of-document token."},
    )
    eod_mask_loss: bool = field(
        default=False, metadata={"help": "Mask loss for the end of document tokens."}
    )

    positional_embedding: PositionalEmbeddingConfig = field(
        default_factory=PositionalEmbeddingConfig
    )

    add_pooler: bool = field(default=True, metadata={"help": "add pooler"})
    untie_embed: bool = field(default=False, metadata={
                              "help": "untie lm head"})
    no_bias: bool = field(
        default=False, metadata={"help": "no bias in layers"}
    )

    fp32_residual_connection: bool = field(
        default=False, metadata={"help": "fp32 residual connection"}
    )
    fp16_lm_cross_entropy: bool = field(
        default=False, metadata={"help": "use fp16 precision in cross entropy"}
    )
    ln_type: str = field(default="layernorm", metadata={
                         "help": "layernorm type"})
    ln_eps: float = field(default=1e-5, metadata={"help": "layernorm epsilon"})
    post_ln: bool = field(default=False, metadata={
                          "help": "use post layer norm"})

    # attention attributes
    head_dim: int = field(default=128, metadata={
                          "help": "attention head dimension"})
    num_attention_heads: Optional[int] = field(
        default=None, metadata={"help": "number of attention head"}
    )
    seq_len_q: Optional[int] = field(default=None, metadata={
                           "help": "query sequence length"})
    seq_len_kv: Optional[int] = field(
        default=None, metadata={"help": "key/value sequence length"}
    )
    num_attention_groups: Optional[int] = field(
        default=None,
        metadata={"help": "number of key-value groups in grouped query attention"},
    )
    attention_dropout: float = field(
        default=0.1, metadata={"help": "dropout ratio in attention"}
    )

    activation_type: str = field(default="gelu", metadata={
                                 "help": "activation type"})

    # HuggingFace compatibility
    hf_model_type: Optional[str] = field(
        default=None,
        metadata={"help": "HuggingFace model_type for checkpoint compatibility"}
    )
    hf_architecture: Optional[str] = field(
        default=None,
        metadata={"help": "HuggingFace architecture name (e.g., 'LlamaForCausalLM')"}
    )

    # tokenizer
    tokenizer_type: str = field(
        default="gpt2",
        metadata={"help": "tokenizer type like bbpe, spe or model: gpt2, llama"},
    )
    vocab_name_or_path: str = field(
        default="gpt2", metadata={"help": "vocab name or path"}
    )
    merge_file_path: Optional[str] = field(
        default=None, metadata={"help": "merge file path"}
    )

    def __post_init__(self):
        if self.ln_type not in ["layernorm", "rmsnorm"]:
            raise ValueError(f"Invalid layer norm type: {self.ln_type}")
