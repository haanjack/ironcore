# Copyright (c) 2025-2026 Jaegeun Han
#
# SPDX-License-Identifier: MIT
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the above copyright notice,
# this list of conditions, and the following disclaimer are retained.
#
# Full license text is available at LICENSE file.

import torch
from torch import nn

try:
    from flash_attn import flash_attn_varlen_func
except ImportError:
    flash_attn_varlen_func = None
from einops import rearrange

from ironcore.config import MainConfig
from ironcore.layers.module import BaseModule
from ironcore.utils import get_model_dtype, profile_context


class Attention(BaseModule):
    """
    Transformer Attention (computation only, no projections)

    This class handles the core attention computation without QKV projections.
    QKV projections should be handled by the model layer for flexibility.

    h: hidden_size
    s: sequence_length
    hn: head numbers
    hd: head dimension

    Expected input shapes:
        query: [b, sq, hn, hd]
        key: [b, sk, gn, hd]
        value: [b, sk, gn, hd]

    """

    def __init__(self, config: MainConfig):
        super().__init__(config)

        # global attention info
        self.num_attention_heads = config.model.num_attention_heads
        self.tensor_model_parallel_size = config.trainer.tensor_model_parallel_size

        self.head_dimension = config.model.d_model // self.num_attention_heads

        # tensor parallel attention info
        self.num_local_attention_heads = (
            self.num_attention_heads // config.trainer.tensor_model_parallel_size
        )
        self.num_local_attention_groups = (
            config.model.num_attention_groups
            // config.trainer.tensor_model_parallel_size
        )

        self.softmax = torch.nn.Softmax(dim=-1)
        self.attn_dropout = nn.Dropout(config.model.dropout_attn)

        self.scale_factor = self.head_dimension**0.5  # attention scale factor
        self.mask_value = torch.finfo(get_model_dtype(self.config)).min

    def _attention(
        self,
        query,
        key,
        value,
        seq_len_q,
        seq_len_kv,
        attention_mask,
    ):
        """Standard attention implementation.

        Args:
            query: [b, sq, hn, hd]
            key: [b, sk, gn, hd]
            value: [b, sk, gn, hd]
        """
        batch_size = key.size(0)

        # query: [b, sq, hn, hd] -> [b, hn, sq, hd]
        # key: [b, sk, gn, hd] -> [b, gn, hd, sk]
        # value: [b, sk, gn, hd] -> [b, gn, sk, hd]
        query = query.transpose(1, 2)
        key = key.permute(0, 2, 3, 1)
        value = value.transpose(1, 2)

        # GQA/MQA support: replicate key/value groups to match query heads
        # key: [b, gn, hd, sk], value: [b, gn, sk, hd]
        if self.num_local_attention_groups != self.num_local_attention_heads:
            # replicate key/value to match with query layer
            key = key.repeat_interleave(
                self.num_local_attention_heads // self.num_local_attention_groups, dim=1
            )
            value = value.repeat_interleave(
                self.num_local_attention_heads // self.num_local_attention_groups, dim=1
            )

        with profile_context("self attention"):
            # attention operation
            # [b, hn, sq, hd] * [b, gn, hd, sk] -> [b, hn, sq, sk]
            attention_score = torch.matmul(query, key)

        with profile_context("self attention"):
            attention_score = attention_score / self.scale_factor
            attention_score = attention_score.view(
                batch_size, self.num_local_attention_heads, seq_len_q, seq_len_kv
            )

            if attention_mask is not None:
                attention_score = attention_score.masked_fill(
                    attention_mask == 0, self.mask_value)

        # max subtraction trick for numerical stability
        with profile_context("attention softmax"):
            max_scores = attention_score.max(dim=-1, keepdim=True)[0]
            attention_score = attention_score - max_scores

            attention_probs = self.softmax(attention_score)

        # dropout
        with profile_context("self attention dropout"):
            if self.config.model.dropout_attn > 0.0:
                attention_probs = self.attn_dropout(attention_probs)

        # attention_probs: [b, hn, sq, sk]
        # value_layer: [b, hn, sk, hd]
        with profile_context("self attention matmul"):
            context_output = torch.matmul(attention_probs, value)

        # context_output: [b, hn, sq, hd] -> [b, sq, hn, hd] -> [b, sq, hn * hd]
        context_output = (
            context_output.transpose(1, 2)
            .reshape(batch_size, seq_len_q, -1)
        )

        return context_output

    def _flash_attention(
        self,
        query,
        key,
        value,
        seq_len_q,
        seq_len_kv,
        max_seqlen_q,
        max_seqlen_k,
        causal=False,
        window_size=(-1, -1),  # -1 means infinite context window
        alibi_slopes=None,
    ):
        """Flash attention implementation.

        Args:
            query: [b, sq, hn, hd]
            key: [b, sk, gn, hd]
            value: [b, sk, gn, hd]
        """
        batch_size = query.size(0)

        query, key, value = [
            x.reshape(-1, self.num_local_attention_heads, self.head_dimension)
            for x in [query, key, value]
        ]
        cu_seqlens_q = torch.arange(
            0,
            (batch_size + 1) * seq_len_q,
            step=seq_len_q,
            dtype=torch.int32,
            device=self.device,
        )
        cu_seqlens_k = torch.arange(
            0,
            (batch_size + 1) * seq_len_kv,
            step=seq_len_kv,
            dtype=torch.int32,
            device=self.device,
        )

        max_seqlen_q = torch.tensor(max_seqlen_q, dtype=torch.int32)
        max_seqlen_k = torch.tensor(max_seqlen_k, dtype=torch.int32)

        # output: [b, sq, hn, hd]
        context_output = flash_attn_varlen_func(
            query,
            key,
            value,
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q,
            max_seqlen_k,
            self.config.model.dropout_attn,
            causal=causal,
            window_size=window_size,
            alibi_slopes=alibi_slopes,
        )

        # output: [b * sq, hn, hd] -> [b, sq, hn * hd]
        context_output = rearrange(context_output, "(b s) h d -> b s (h d)", b=batch_size)

        return context_output

    def forward(
        self,
        query,
        key,
        value,
        attention_mask=None,
    ):
        """
        Compute attention given pre-projected Q, K, V tensors.

        Args:
            query: [b, sq, hn, hd] - Query tensor (already projected and with RoPE if applicable)
            key: [b, sk, gn, hd] - Key tensor (already projected and with RoPE if applicable)
            value: [b, sk, gn, hd] - Value tensor (already projected)
            attention_mask: Optional attention mask

        Returns:
            context_output: [b, sq, hn * hd]
        """
        seq_len_q = query.size(1)
        seq_len_kv = key.size(1)

        if not self.config.trainer.use_flash_attn or flash_attn_varlen_func is None:
            context_output = self._attention(
                query,
                key,
                value,
                seq_len_q,
                seq_len_kv,
                attention_mask,
            )
        else:
            context_output = self._flash_attention(
                query,
                key,
                value,
                seq_len_q,
                seq_len_kv,
                seq_len_q,
                seq_len_kv,
                causal=True,
                window_size=(-1, -1),
            )

        # output: [b, sq, hn * hd]
        return context_output
