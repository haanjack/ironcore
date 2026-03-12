# Copyright (c) 2025-2026 Jaegeun Han
#
# SPDX-License-Identifier: Apache-2.0

import torch
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
            config.model.num_attention_groups // config.trainer.tensor_model_parallel_size
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
        # GQA/MQA support: replicate key/value groups to match query heads
        if key.size(2) != query.size(2):
            key = expand_for_gqa(
                key, self.num_local_attention_groups, self.num_local_attention_heads, kv_dim=2
            )
            value = expand_for_gqa(
                value, self.num_local_attention_groups, self.num_local_attention_heads, kv_dim=2
            )

        with profile_context("self attention"):
            # attention operation using einsum: [b, hn, sq, hd] * [b, hn, sk, hd] -> [b, hn, sq, sk]
            # query: [b, sq, hn, hd], key: [b, sk, hn, hd]
            if query.dtype != key.dtype:
                print(f"DTYPE MISMATCH: query={query.dtype}, key={key.dtype}")
            attention_score = torch.einsum("bqnd,bknd->bnqk", query, key)

        with profile_context("self attention scale"):
            attention_score = attention_score / self.scale_factor

            if attention_mask is not None:
                # attention_mask: [b, 1, sq, sk]
                attention_score = attention_score.masked_fill(attention_mask == 0, self.mask_value)

        # max subtraction trick for numerical stability
        with profile_context("attention softmax"):
            # Cast to fp32 for stable softmax
            attention_probs = torch.softmax(attention_score.float(), dim=-1).to(query.dtype)

        # dropout
        with profile_context("self attention dropout"):
            if self.config.model.dropout_attn > 0.0:
                attention_probs = self.attn_dropout(attention_probs)

        # attention_probs: [b, hn, sq, sk], value: [b, sk, hn, hd]
        with profile_context("self attention matmul"):
            # [b, hn, sq, sk] * [b, sk, hn, hd] -> [b, sq, hn, hd]
            context_output = torch.einsum("bnqk,bknd->bqnd", attention_probs, value)

        # context_output: [b, sq, hn, hd] -> [b, sq, hn * hd]
        context_output = rearrange(context_output, "b s n d -> b s (n d)")

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
            device=query.device,
        )
        cu_seqlens_k = torch.arange(
            0,
            (batch_size + 1) * seq_len_kv,
            step=seq_len_kv,
            dtype=torch.int32,
            device=key.device,
        )

        max_seqlen_q = torch.tensor(max_seqlen_q, dtype=torch.int32)
        max_seqlen_k = torch.tensor(max_seqlen_k, dtype=torch.int32)

        # output: [b, sq, hn, hd]
        context_output = flash_attn_varlen_func(  # type: ignore
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
        use_cache=False,
        past_kv=None,
    ):
        """
        Compute attention given pre-projected Q, K, V tensors.

        Args:
            query: [b, sq, hn, hd] - Query tensor (already projected and with RoPE if applicable)
            key: [b, sk, gn, hd] - Key tensor (already projected and with RoPE if applicable)
            value: [b, sk, gn, hd] - Value tensor (already projected)
            attention_mask: Optional attention mask
            use_cache: Whether to use KV cache
            past_kv: Optional tuple of (past_key, past_value) from cache

        Returns:
            If use_cache: (context_output, (key, value))
            Otherwise: context_output
            context_output: [b, sq, hn * hd]
        """
        # Handle cached KV
        if use_cache and past_kv is not None:
            past_key, past_value = past_kv
            # Concatenate cached KV with new KV
            # past_key/value: [b, past_len, gn, hd]
            # key/value: [b, new_len, gn, hd]
            key = torch.cat([past_key, key], dim=1)
            value = torch.cat([past_value, value], dim=1)

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
        if use_cache:
            return context_output, (key, value)
        return context_output

    def _flash_attention_with_cache(
        self,
        query,
        key_cache,
        value_cache,
        cache_seqlens,
        k_new=None,
        v_new=None,
        cache_batch_idx=None,
    ):
        """Flash attention with pre-allocated KV cache.

        Uses flash_attn_with_kvcache for efficient inference with cache.
        This bypasses the need to concatenate past and current KV tensors.

        Args:
            query: [b, sq, hn, hd] - Query tensor
            key_cache: [b, hn, max_seq, hd] - Pre-allocated key cache
            value_cache: [b, hn, max_seq, hd] - Pre-allocated value cache
            cache_seqlens: [b] - Current sequence lengths in cache
            k_new: [b, sq, hn, hd] - New keys to append (optional)
            v_new: [b, sq, hn, hd] - New values to append (optional)
            cache_batch_idx: Optional batch indices for selective update

        Returns:
            context_output: [b, sq, hn * hd]
        """
        try:
            from flash_attn import flash_attn_with_kvcache
        except ImportError:
            raise RuntimeError(
                "flash_attn_with_kvcache not available. "
                "Install flash-attn>=2.5.0 or use standard attention."
            )

        batch_size = query.size(0)
        seq_len_q = query.size(1)

        # Call flash_attn_with_kvcache
        # The function handles cache updates internally
        context_output = flash_attn_with_kvcache(
            q=query,
            k_cache=key_cache,
            v_cache=value_cache,
            cache_seqlens=cache_seqlens,
            k=k_new,
            v=v_new,
            cache_batch_idx=cache_batch_idx,
            causal=True,
            softmax_scale=self.scale_factor,
        )

        # output: [b, sq, hn, hd] -> [b, sq, hn * hd]
        context_output = context_output.reshape(batch_size, seq_len_q, -1)

        return context_output
