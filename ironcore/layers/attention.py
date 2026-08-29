# Copyright (c) 2025-2026 Jaegeun Han
#
# SPDX-License-Identifier: Apache-2.0

import torch
import torch.nn.functional as F
from torch import nn

try:
    from flash_attn import flash_attn_varlen_func
except ImportError:
    flash_attn_varlen_func = None

from einops import rearrange

from ironcore.config import MainConfig
from ironcore.layers.kv_cache_utils import expand_for_gqa
from ironcore.layers.module import BaseModule
from ironcore.utils import get_model_dtype, profile_context


class Attention(BaseModule):
    """
    Transformer Attention (computation only, no projections)
    """

    def __init__(self, config: MainConfig):
        super().__init__(config)

        self.num_attention_heads = config.model.num_attention_heads
        self.tensor_model_parallel_size = config.trainer.tensor_model_parallel_size
        self.head_dimension = config.model.head_dim

        self.num_local_attention_heads = (
            self.num_attention_heads // config.trainer.tensor_model_parallel_size
        )
        self.num_local_attention_groups = (
            config.model.num_attention_groups // config.trainer.tensor_model_parallel_size
        )

        self.softmax = torch.nn.Softmax(dim=-1)
        self.attn_dropout = nn.Dropout(config.model.dropout_attn)

        self.scale_factor = self.head_dimension**0.5
        self.mask_value = torch.finfo(get_model_dtype(self.config)).min

    def _attention(
        self,
        query,
        key,
        value,
        attention_mask,
        is_causal: bool = False,
    ):
        """Attention via F.scaled_dot_product_attention using [b, s, n, d] layout."""
        # GQA expansion
        if key.size(2) != query.size(2):
            key = expand_for_gqa(
                key, self.num_local_attention_groups, self.num_local_attention_heads, kv_dim=2
            )
            value = expand_for_gqa(
                value, self.num_local_attention_groups, self.num_local_attention_heads, kv_dim=2
            )

        # SDPA expects [b, n, s, d]; contiguous() ensures fast kernel dispatch
        query = query.transpose(1, 2).contiguous()
        key = key.transpose(1, 2).contiguous()
        value = value.transpose(1, 2).contiguous()

        dropout_p = self.config.model.dropout_attn if self.training else 0.0
        # is_causal=True is the reliable path under torch.compile — SDPA boolean masks
        # can be silently ignored by the inductor backend, producing non-causal attention.
        # Use the explicit mask only for inference with non-square (q_len != kv_len) attention.
        if is_causal:
            sdpa_mask = None
        else:
            sdpa_mask = attention_mask.bool() if attention_mask is not None else None

        with profile_context("self attention"):
            context_output = F.scaled_dot_product_attention(
                query,
                key,
                value,
                attn_mask=sdpa_mask,
                dropout_p=dropout_p,
                is_causal=is_causal,
            )

        # [b, n, sq, d] -> [b, sq, n*d]
        context_output = context_output.transpose(1, 2)
        context_output = rearrange(context_output, "b q n d -> b q (n d)")
        return context_output

    def _flash_attention(
        self,
        query,
        key,
        value,
        seq_len_q,
        seq_len_kv,
        causal=True,
    ):
        """Flash attention implementation using flash_attn_varlen_func.

        Args:
            query: [b, sq, hn, hd]
            key:   [b, sk, gn, hd]  (gn = num_local_attention_groups for GQA)
            value: [b, sk, gn, hd]
        """
        batch_size = query.size(0)
        if seq_len_kv <= 0:
            return query.new_zeros(
                batch_size, query.size(1), query.size(2), self.head_dimension, device=query.device
            )

        # Flatten batch+seq: [b*sq, hn, hd] / [b*sk, gn, hd]
        query = query.reshape(-1, self.num_local_attention_heads, self.head_dimension)
        key = key.reshape(-1, self.num_local_attention_groups, self.head_dimension)
        value = value.reshape(-1, self.num_local_attention_groups, self.head_dimension)

        # Optimization: Cache these if batch_size and seq_len are constant
        cu_seqlens_q = torch.arange(
            0, (batch_size + 1) * seq_len_q, step=seq_len_q, dtype=torch.int32, device=query.device
        )
        cu_seqlens_k = torch.arange(
            0, (batch_size + 1) * seq_len_kv, step=seq_len_kv, dtype=torch.int32, device=key.device
        )

        # Gate dropout on training mode — matches the SDPA path. Flash attention
        # was passing dropout unconditionally, applying it during eval/inference.
        dropout_p = self.config.model.dropout_attn if self.training else 0.0

        context_output = flash_attn_varlen_func(  # type: ignore
            query,
            key,
            value,
            cu_seqlens_q,
            cu_seqlens_k,
            seq_len_q,
            seq_len_kv,
            dropout_p,
            causal=causal,
        )

        # [b*sq, hn, hd] -> [b, sq, hn*hd]
        context_output = rearrange(context_output, "(b s) h d -> b s (h d)", b=batch_size)
        return context_output

    def forward(
        self,
        query,
        key,
        value,
        attention_mask=None,
        use_cache=False,
        past_kv=None,
    ):
        # Concatenate if using functional cache
        if use_cache and past_kv is not None:
            past_key, past_value = past_kv
            key = torch.cat([past_key, key], dim=1)
            value = torch.cat([past_value, value], dim=1)

        seq_len_q = query.size(1)
        seq_len_kv = key.size(1)

        if self.config.trainer.use_flash_attn and flash_attn_varlen_func is not None:
            context_output = self._flash_attention(query, key, value, seq_len_q, seq_len_kv)
        else:
            # Full-sequence prefill (training or non-cached inference): always causal.
            # Decode step (use_cache + past_kv present, or q_len < kv_len): use explicit mask.
            is_causal = seq_len_q == seq_len_kv and not use_cache
            context_output = self._attention(query, key, value, attention_mask, is_causal=is_causal)

        if use_cache:
            return context_output, (key, value)
        return context_output
