# Copyright (c) 2025-2026 Jaegeun Han
#
# SPDX-License-Identifier: Apache-2.0

import torch
from torch import nn

try:
    from flash_attn import flash_attn_varlen_func
except ImportError:
    flash_attn_varlen_func = None


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
    ):
        """Standard attention implementation using [b, s, n, d] layout."""
        # GQA expansion
        if key.size(2) != query.size(2):
            key = expand_for_gqa(
                key, self.num_local_attention_groups, self.num_local_attention_heads, kv_dim=2
            )
            value = expand_for_gqa(
                value, self.num_local_attention_groups, self.num_local_attention_heads, kv_dim=2
            )

        with profile_context("self attention score"):
            # query: [b, sq, n, d], key: [b, sk, n, d] -> [b, n, sq, sk]
            attention_score = torch.einsum("bqnd,bknd->bnqk", query, key)
            attention_score = attention_score / self.scale_factor

            if attention_mask is not None:
                attention_score = attention_score.masked_fill(attention_mask == 0, self.mask_value)

        # Softmax in fp32
        with profile_context("attention softmax"):
            attention_probs = self.softmax(attention_score.float()).to(query.dtype)

        if self.config.model.dropout_attn > 0.0:
            attention_probs = self.attn_dropout(attention_probs)

        # Matmul: [b, n, sq, sk] * [b, sk, n, d] -> [b, sq, n, d]
        with profile_context("self attention context"):
            context_output = torch.einsum("bnqk,bknd->bqnd", attention_probs, value)

        # Reshape to [b, sq, n*d]
        from einops import rearrange
        context_output = rearrange(context_output, "b q n d -> b q (n d)")
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

        # Use standard attention path
        context_output = self._attention(
            query,
            key,
            value,
            attention_mask,
        )

        if use_cache:
            return context_output, (key, value)
        return context_output
