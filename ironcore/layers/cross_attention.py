# Copyright (c) 2025-2026 Jaegeun Han
#
# SPDX-License-Identifier: MIT
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the above copyright notice,
# this list of conditions, and the following disclaimer are retained.
#
# Full license text is available at LICENSE file.

"""Cross-attention layers for vision-language fusion."""

import torch
import torch.nn.functional as F
from torch import nn

from ironcore.config import MainConfig
from ironcore.layers.module import BaseModule
from ironcore.parallel.tensor_parallel import ColumnParallelLinear, RowParallelLinear


class CrossAttention(BaseModule):
    """Cross-attention for vision-language fusion.

    Language tokens (queries) attend to vision tokens (keys/values).
    Uses the same patterns as IronCore's self-attention but with
    separate key/value inputs from vision encoder.

    Args:
        config: MainConfig
        hidden_size: Language model hidden size
        num_heads: Number of attention heads
        kv_hidden_size: Vision encoder hidden size (for cross attention)
    """

    def __init__(
        self,
        config: MainConfig,
        hidden_size: int,
        num_heads: int,
        kv_hidden_size: int | None = None,
    ):
        super().__init__(config)

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.scale = self.head_dim ** -0.5

        # KV dimension (from vision encoder)
        self.kv_hidden_size = kv_hidden_size or hidden_size

        # TP settings
        self.tp_size = config.trainer.tensor_model_parallel_size
        self.num_local_heads = num_heads // self.tp_size

        # Query projection (from language hidden states)
        self.q_proj = ColumnParallelLinear(
            config,
            hidden_size,
            hidden_size,
            bias=False,
            gather_output=False,
        )

        # Key/Value projections (from vision features)
        # Note: These may need different dimensions if vision_hidden != language_hidden
        self.k_proj = ColumnParallelLinear(
            config,
            self.kv_hidden_size,
            hidden_size,
            bias=False,
            gather_output=False,
        )
        self.v_proj = ColumnParallelLinear(
            config,
            self.kv_hidden_size,
            hidden_size,
            bias=False,
            gather_output=False,
        )

        # Output projection
        self.out_proj = RowParallelLinear(
            config,
            hidden_size,
            hidden_size,
            bias=False,
            input_is_parallel=True,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Cross-attention forward pass.

        Args:
            hidden_states: [batch, seq_len, hidden_size] language hidden states (queries)
            encoder_hidden_states: [batch, vision_len, kv_hidden_size] vision features (keys/values)
            attention_mask: Optional mask for vision tokens

        Returns:
            [batch, seq_len, hidden_size] attended hidden states
        """
        batch_size, seq_len, _ = hidden_states.shape
        vision_len = encoder_hidden_states.shape[1]

        # Project queries from language hidden states
        q = self.q_proj(hidden_states)  # [batch, seq_len, hidden_size]

        # Project keys and values from vision features
        k = self.k_proj(encoder_hidden_states)  # [batch, vision_len, hidden_size]
        v = self.v_proj(encoder_hidden_states)  # [batch, vision_len, hidden_size]

        # Reshape for multi-head attention
        # [batch, seq, hidden] -> [batch, seq, num_local_heads, head_dim]
        q = q.view(batch_size, seq_len, self.num_local_heads, self.head_dim)
        k = k.view(batch_size, vision_len, self.num_local_heads, self.head_dim)
        v = v.view(batch_size, vision_len, self.num_local_heads, self.head_dim)

        # Transpose for attention: [batch, num_heads, seq, head_dim]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Compute attention scores
        # [batch, num_heads, seq_q, head_dim] @ [batch, num_heads, head_dim, seq_k]
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        # Apply attention mask if provided
        if attention_mask is not None:
            # attention_mask: [batch, vision_len] -> [batch, 1, 1, vision_len]
            if attention_mask.dim() == 2:
                attention_mask = attention_mask[:, None, None, :]
            attn_weights = attn_weights.masked_fill(
                attention_mask == 0,
                torch.finfo(attn_weights.dtype).min,
            )

        # Softmax
        attn_weights = F.softmax(attn_weights, dim=-1)

        # Apply attention to values
        # [batch, num_heads, seq_q, seq_k] @ [batch, num_heads, seq_k, head_dim]
        attn_output = torch.matmul(attn_weights, v)

        # Reshape back: [batch, num_heads, seq, head_dim] -> [batch, seq, hidden]
        attn_output = attn_output.transpose(1, 2).reshape(batch_size, seq_len, -1)

        # Output projection
        output = self.out_proj(attn_output)

        return output


class GatedCrossAttention(BaseModule):
    """Gated Cross-Attention Layer (Flamingo-style).

    Combines cross-attention with a gating mechanism for stable training.
    Architecture: LayerNorm -> Cross-Attention -> Gated Residual -> LayerNorm -> FFN

    The gate starts at 0 and learns when to incorporate vision information.

    Args:
        config: MainConfig
        hidden_size: Language model hidden size
        ffn_hidden_size: FFN intermediate size
        num_heads: Number of attention heads
        kv_hidden_size: Vision encoder hidden size
    """

    def __init__(
        self,
        config: MainConfig,
        hidden_size: int,
        ffn_hidden_size: int,
        num_heads: int,
        kv_hidden_size: int | None = None,
    ):
        super().__init__(config)

        self.hidden_size = hidden_size

        # Layer norms
        self.norm_before_ca = nn.LayerNorm(hidden_size, eps=config.model.ln_eps)
        self.norm_before_ffn = nn.LayerNorm(hidden_size, eps=config.model.ln_eps)

        # Cross-attention
        self.cross_attn = CrossAttention(
            config,
            hidden_size=hidden_size,
            num_heads=num_heads,
            kv_hidden_size=kv_hidden_size,
        )

        # Learnable gate parameter (starts at 0 for stable training)
        self.gate = nn.Parameter(torch.zeros(1))

        # FFN (same pattern as IronCore's MLP)
        self.ffn_up = ColumnParallelLinear(
            config,
            hidden_size,
            ffn_hidden_size,
            bias=False,
            gather_output=False,
        )
        self.ffn_down = RowParallelLinear(
            config,
            ffn_hidden_size,
            hidden_size,
            bias=False,
            input_is_parallel=True,
        )
        self.ffn_act = nn.GELU()

        # Dropout
        self.dropout = nn.Dropout(config.model.dropout_attn)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Gated cross-attention forward pass.

        Args:
            hidden_states: [batch, seq_len, hidden_size] language hidden states
            encoder_hidden_states: [batch, vision_len, kv_hidden_size] vision features
            attention_mask: Optional mask for vision tokens

        Returns:
            [batch, seq_len, hidden_size] updated hidden states
        """
        residual = hidden_states

        # Cross-attention with pre-norm
        hidden_states = self.norm_before_ca(hidden_states)
        cross_attn_output = self.cross_attn(
            hidden_states,
            encoder_hidden_states,
            attention_mask,
        )

        # Gated residual connection
        # gate starts at 0, gradually learns to incorporate vision
        hidden_states = residual + self.dropout(cross_attn_output) * self.gate.tanh()

        # FFN with pre-norm
        residual = hidden_states
        hidden_states = self.norm_before_ffn(hidden_states)

        ffn_output = self.ffn_up(hidden_states)
        ffn_output = self.ffn_act(ffn_output)
        ffn_output = self.ffn_down(ffn_output)

        hidden_states = residual + self.dropout(ffn_output)

        return hidden_states


class VisionLanguageFusion(BaseModule):
    """Vision-Language Fusion Module.

    Provides multiple fusion strategies:
    - "gated_cross_attention": Flamingo-style gated cross-attention layers
    - "simple_concat": Simple concatenation (baseline)
    - "qformer": BLIP-2 style Q-Former

    Args:
        config: MainConfig
        num_layers: Number of cross-attention layers
        fusion_type: Type of fusion to use
    """

    def __init__(
        self,
        config: MainConfig,
        num_layers: int = 1,
        fusion_type: str = "gated_cross_attention",
    ):
        super().__init__(config)

        self.fusion_type = fusion_type
        self.num_layers = num_layers

        language_hidden = config.model.d_model
        ffn_hidden = config.model.d_ffn
        num_heads = config.model.num_attention_heads

        if fusion_type == "gated_cross_attention":
            self.layers = nn.ModuleList([
                GatedCrossAttention(
                    config,
                    hidden_size=language_hidden,
                    ffn_hidden_size=ffn_hidden,
                    num_heads=num_heads,
                    kv_hidden_size=language_hidden,  # Use language_hidden since projector already converts
                )
                for _ in range(num_layers)
            ])

        elif fusion_type == "qformer":
            # Q-Former: learnable queries that attend to vision
            self.query_tokens = nn.Parameter(
                torch.randn(1, config.vla.num_image_tokens, language_hidden)
            )
            self.cross_attn = CrossAttention(
                config,
                hidden_size=language_hidden,
                num_heads=num_heads,
                kv_hidden_size=language_hidden,  # Use language_hidden since projector already converts
            )

        else:
            raise ValueError(f"Unknown fusion type: {fusion_type}")

    def forward(
        self,
        hidden_states: torch.Tensor,
        vision_features: torch.Tensor,
        vision_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Fuse vision and language features.

        Args:
            hidden_states: [batch, seq_len, hidden_size] language hidden states
            vision_features: [batch, vision_len, vision_hidden] vision features
            vision_mask: Optional mask for vision tokens

        Returns:
            [batch, seq_len, hidden_size] fused hidden states
        """
        if self.fusion_type == "gated_cross_attention":
            for layer in self.layers:
                hidden_states = layer(
                    hidden_states,
                    vision_features,
                    vision_mask,
                )
            return hidden_states

        elif self.fusion_type == "qformer":
            batch_size = hidden_states.size(0)

            # Expand query tokens for batch
            query_tokens = self.query_tokens.expand(batch_size, -1, -1)

            # Cross-attend to vision features
            fused_queries = self.cross_attn(
                query_tokens,
                vision_features,
                vision_mask,
            )

            # Concatenate fused queries with language hidden states
            return torch.cat([fused_queries, hidden_states], dim=1)

        return hidden_states


__all__ = [
    "CrossAttention",
    "GatedCrossAttention",
    "VisionLanguageFusion",
]
