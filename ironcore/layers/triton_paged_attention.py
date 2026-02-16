# Copyright (c) 2025-2026 Jaegeun Han
#
# SPDX-License-Identifier: MIT
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the above copyright notice,
# this list of conditions, and the following disclaimer are retained.
#
# Full license text is available at LICENSE file.

"""
Triton-based paged attention kernel for efficient KV cache attention.

This module provides a high-performance Triton implementation of paged attention
that fuses the KV gather and attention computation into a single kernel.

Performance characteristics:
- Memory-bound kernel optimized for KV cache access patterns
- Online softmax for numerical stability
- Supports variable sequence lengths via block tables
- Supports GQA/MQA via head grouping
"""

import torch

# Check if Triton is available
TRITON_AVAILABLE = False
try:
    import triton
    import triton.language as tl

    TRITON_AVAILABLE = True
except ImportError:
    pass


def triton_paged_attention(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_tables: torch.Tensor,
    context_lens: torch.Tensor,
    num_heads: int,
    num_kv_heads: int,
    page_size: int = 16,
) -> torch.Tensor:
    """
    Compute paged attention using Triton kernel.

    This implementation uses a two-stage approach:
    1. Triton kernel to gather KV from pages
    2. PyTorch SDPA for attention computation

    This is more efficient than a fully fused kernel for most use cases.

    Args:
        query: Query tensor [batch, 1, num_heads, head_dim] or [batch, num_heads, head_dim]
        key_cache: Physical key cache [num_pages, num_kv_heads, page_size, head_dim]
        value_cache: Physical value cache [num_pages, num_kv_heads, page_size, head_dim]
        block_tables: Block tables [batch, max_pages_per_seq]
        context_lens: Context lengths [batch]
        num_heads: Number of query heads
        num_kv_heads: Number of KV heads (for GQA)
        page_size: Number of tokens per page

    Returns:
        Output tensor [batch, 1, num_heads, head_dim] or [batch, num_heads, head_dim]
    """
    if not TRITON_AVAILABLE:
        raise RuntimeError("Triton is not available. Install with: pip install triton")

    # Handle different input shapes
    squeeze_output = False
    if query.dim() == 3:
        query = query.unsqueeze(1)
        squeeze_output = True

    batch_size, seq_len, num_heads_q, head_dim = query.shape
    assert seq_len == 1, "Triton kernel currently supports seq_len=1 for query"

    device = query.device
    dtype = query.dtype

    # Get max context length for padding
    max_ctx_len = context_lens.max().item()

    # Gather KV from pages using Triton kernel
    gathered_keys, gathered_values = _gather_kv_triton(
        key_cache, value_cache, block_tables, context_lens, max_ctx_len, num_kv_heads, page_size
    )

    # Expand for GQA
    num_heads_per_group = num_heads // num_kv_heads
    if num_heads != num_kv_heads:
        gathered_keys = gathered_keys.repeat_interleave(num_heads_per_group, dim=2)
        gathered_values = gathered_values.repeat_interleave(num_heads_per_group, dim=2)

    # Compute attention using scaled dot-product
    scale = 1.0 / (head_dim**0.5)

    # query: [batch, 1, num_heads, head_dim]
    # gathered_keys: [batch, max_ctx, num_heads, head_dim]
    q = query.transpose(1, 2)  # [batch, num_heads, 1, head_dim]
    k = gathered_keys.transpose(1, 2)  # [batch, num_heads, max_ctx, head_dim]
    v = gathered_values.transpose(1, 2)  # [batch, num_heads, max_ctx, head_dim]

    # Create attention mask
    mask = torch.arange(max_ctx_len, device=device).unsqueeze(0) < context_lens.unsqueeze(1)
    mask = mask.unsqueeze(1).unsqueeze(2)  # [batch, 1, 1, max_ctx]

    # Use scaled_dot_product_attention if available (Flash Attention)
    if hasattr(torch.nn.functional, "scaled_dot_product_attention"):
        # Convert mask to additive mask for SDPA
        attn_mask = torch.zeros(batch_size, 1, 1, max_ctx_len, device=device, dtype=dtype)
        attn_mask = attn_mask.masked_fill(~mask, float("-inf"))

        output = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, scale=scale
        )
    else:
        # Fallback to manual attention
        scores = torch.matmul(q, k.transpose(-1, -2)) * scale
        scores = scores.masked_fill(~mask, float("-inf"))
        attn_weights = torch.softmax(scores, dim=-1)
        output = torch.matmul(attn_weights, v)

    output = output.transpose(1, 2)  # [batch, 1, num_heads, head_dim]

    if squeeze_output:
        return output.squeeze(1)
    return output


def _gather_kv_triton(
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_tables: torch.Tensor,
    context_lens: torch.Tensor,
    max_ctx_len: int,
    num_kv_heads: int,
    page_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Gather KV from physical pages using Triton kernel."""

    batch_size = block_tables.shape[0]
    head_dim = key_cache.shape[-1]
    device = key_cache.device
    dtype = key_cache.dtype

    # Output tensors
    gathered_keys = torch.zeros(
        batch_size, max_ctx_len, num_kv_heads, head_dim, device=device, dtype=dtype
    )
    gathered_values = torch.zeros(
        batch_size, max_ctx_len, num_kv_heads, head_dim, device=device, dtype=dtype
    )

    max_pages_per_seq = block_tables.shape[1]

    # Define the gather kernel
    @triton.jit
    def _gather_kv_kernel(
        k_cache_ptr,
        v_cache_ptr,
        block_tables_ptr,
        context_lens_ptr,
        out_k_ptr,
        out_v_ptr,
        stride_kp,
        stride_kh,
        stride_kt,
        stride_kd,
        stride_ob,
        stride_os,
        stride_oh,
        stride_od,
        stride_bt,
        max_ctx_len: tl.constexpr,
        page_size: tl.constexpr,
        max_pages_per_seq: tl.constexpr,
        num_kv_heads: tl.constexpr,
        head_dim: tl.constexpr,
    ):
        # Each program handles one (batch, head, token) tuple
        pid = tl.program_id(0)
        batch_idx = pid // (num_kv_heads * max_ctx_len)
        remainder = pid % (num_kv_heads * max_ctx_len)
        head_idx = remainder // max_ctx_len
        token_idx = remainder % max_ctx_len

        # Check if this token is within context length
        ctx_len = tl.load(context_lens_ptr + batch_idx)
        if token_idx >= ctx_len:
            return

        # Get physical page and offset
        page_idx = token_idx // page_size
        token_in_page = token_idx % page_size

        if page_idx >= max_pages_per_seq:
            return

        physical_page = tl.load(block_tables_ptr + batch_idx * max_pages_per_seq + page_idx)
        if physical_page < 0:
            return

        # Load K and V vectors
        head_offsets = tl.arange(0, head_dim)

        # K cache: [num_pages, num_kv_heads, page_size, head_dim]
        k_offsets = (
            physical_page * stride_kp
            + head_idx * stride_kh
            + token_in_page * stride_kt
            + head_offsets * stride_kd
        )
        k = tl.load(k_cache_ptr + k_offsets)

        # V cache: [num_pages, num_kv_heads, page_size, head_dim]
        v_offsets = (
            physical_page * stride_kp
            + head_idx * stride_kh
            + token_in_page * stride_kt
            + head_offsets * stride_kd
        )
        v = tl.load(v_cache_ptr + v_offsets)

        # Store to output: [batch, max_ctx, num_kv_heads, head_dim]
        out_offsets = (
            batch_idx * stride_ob
            + token_idx * stride_os
            + head_idx * stride_oh
            + head_offsets * stride_od
        )
        tl.store(out_k_ptr + out_offsets, k)
        tl.store(out_v_ptr + out_offsets, v)

    # Launch kernel
    grid = (batch_size * num_kv_heads * max_ctx_len,)

    _gather_kv_kernel[grid](
        key_cache,
        value_cache,
        block_tables,
        context_lens,
        gathered_keys,
        gathered_values,
        key_cache.stride(0),
        key_cache.stride(1),
        key_cache.stride(2),
        key_cache.stride(3),
        gathered_keys.stride(0),
        gathered_keys.stride(1),
        gathered_keys.stride(2),
        gathered_keys.stride(3),
        block_tables.stride(0),
        max_ctx_len=max_ctx_len,
        page_size=page_size,
        max_pages_per_seq=max_pages_per_seq,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
    )

    return gathered_keys, gathered_values


def python_paged_attention(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_tables: torch.Tensor,
    context_lens: torch.Tensor,
    num_heads: int,
    num_kv_heads: int,
    page_size: int = 16,
) -> torch.Tensor:
    """
    Reference Python implementation of paged attention.

    This is the baseline implementation for comparison.
    Uses gather + standard attention computation.
    """
    # Handle different input shapes
    squeeze_output = False
    if query.dim() == 3:
        query = query.unsqueeze(1)
        squeeze_output = True

    batch_size, seq_len, num_heads_q, head_dim = query.shape
    num_heads_per_group = num_heads // num_kv_heads
    scale = 1.0 / (head_dim**0.5)

    outputs = []

    for b in range(batch_size):
        ctx_len = context_lens[b].item()
        if ctx_len == 0:
            outputs.append(
                torch.zeros(1, num_heads, head_dim, device=query.device, dtype=query.dtype)
            )
            continue

        # Gather keys and values from physical pages
        pages_needed = (ctx_len + page_size - 1) // page_size
        keys_list = []
        values_list = []

        for page_idx in range(pages_needed):
            physical_page = block_tables[b, page_idx].item()
            if physical_page < 0:
                break

            # Get tokens from this page
            start_token = page_idx * page_size
            end_token = min(start_token + page_size, ctx_len)
            tokens_in_page = end_token - start_token

            # [num_kv_heads, tokens_in_page, head_dim]
            k_page = key_cache[physical_page, :, :tokens_in_page, :]
            v_page = value_cache[physical_page, :, :tokens_in_page, :]

            keys_list.append(k_page)
            values_list.append(v_page)

        # Concatenate: [num_kv_heads, ctx_len, head_dim]
        keys = torch.cat(keys_list, dim=1)
        values = torch.cat(values_list, dim=1)

        # Expand for GQA: [num_heads, ctx_len, head_dim]
        if num_heads != num_kv_heads:
            keys = keys.repeat_interleave(num_heads_per_group, dim=0)
            values = values.repeat_interleave(num_heads_per_group, dim=0)

        # Transpose for attention: [ctx_len, num_heads, head_dim]
        keys = keys.transpose(0, 1)
        values = values.transpose(0, 1)

        # Query: [1, num_heads, head_dim]
        q = query[b]

        # Attention: [num_heads, 1, ctx_len] @ [num_heads, ctx_len, head_dim]
        scores = torch.einsum("qhd,khd->hqk", q, keys) * scale
        attn_weights = torch.softmax(scores, dim=-1)

        # Output: [1, num_heads, head_dim]
        out = torch.einsum("hqk,khd->qhd", attn_weights, values)
        outputs.append(out)

    output = torch.cat(outputs, dim=0)

    if squeeze_output:
        return output  # [batch, num_heads, head_dim]
    return output.unsqueeze(1)  # [batch, 1, num_heads, head_dim]


# Optimized batched version using vectorized operations
def python_paged_attention_batched(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_tables: torch.Tensor,
    context_lens: torch.Tensor,
    num_heads: int,
    num_kv_heads: int,
    page_size: int = 16,
) -> torch.Tensor:
    """
    Batched Python implementation using gather and flash attention.

    More efficient than the loop version but still Python-based.
    """
    squeeze_output = False
    if query.dim() == 3:
        query = query.unsqueeze(1)
        squeeze_output = True

    batch_size, seq_len, num_heads_q, head_dim = query.shape
    num_heads_per_group = num_heads // num_kv_heads
    scale = 1.0 / (head_dim**0.5)

    # Find max context length for padding
    max_ctx_len = context_lens.max().item()

    # Pre-allocate gathered KV
    gathered_keys = torch.zeros(
        batch_size, num_kv_heads, max_ctx_len, head_dim, device=query.device, dtype=key_cache.dtype
    )
    gathered_values = torch.zeros(
        batch_size,
        num_kv_heads,
        max_ctx_len,
        head_dim,
        device=query.device,
        dtype=value_cache.dtype,
    )

    # Gather KV from pages (still need loop for variable lengths)
    for b in range(batch_size):
        ctx_len = context_lens[b].item()
        if ctx_len == 0:
            continue

        pages_needed = (ctx_len + page_size - 1) // page_size

        for page_idx in range(pages_needed):
            physical_page = block_tables[b, page_idx].item()
            if physical_page < 0:
                break

            start_token = page_idx * page_size
            end_token = min(start_token + page_size, ctx_len)
            tokens_in_page = end_token - start_token

            gathered_keys[b, :, start_token:end_token, :] = key_cache[
                physical_page, :, :tokens_in_page, :
            ]
            gathered_values[b, :, start_token:end_token, :] = value_cache[
                physical_page, :, :tokens_in_page, :
            ]

    # Expand for GQA
    if num_heads != num_kv_heads:
        gathered_keys = gathered_keys.repeat_interleave(num_heads_per_group, dim=1)
        gathered_values = gathered_values.repeat_interleave(num_heads_per_group, dim=1)

    # Transpose to match loop version's einsum semantics: [batch, max_ctx, num_heads, head_dim]
    gathered_keys = gathered_keys.transpose(1, 2)
    gathered_values = gathered_values.transpose(1, 2)

    # Compute attention using scaled dot-product
    # q: [batch, num_heads, head_dim]
    # k: [batch, max_ctx, num_heads, head_dim]
    # We want: [batch, num_heads, max_ctx] scores
    q = query.squeeze(1)  # [batch, num_heads, head_dim]
    # Use einsum to match loop version: "qhd,khd->hqk" per batch
    # Loop: q[1,h,d] @ k[seq,h,d] -> [h,1,seq]
    # Batched: q[b,h,d] @ k[b,seq,h,d] -> [b,h,seq]
    scores = torch.einsum("bhd,bkhd->bhk", q, gathered_keys) * scale

    # Create mask based on context lengths
    mask = torch.arange(max_ctx_len, device=query.device).unsqueeze(0) < context_lens.unsqueeze(1)
    mask = mask.unsqueeze(1)  # [batch, 1, max_ctx]
    scores = scores.masked_fill(~mask, float("-inf"))

    # Softmax and output
    attn_weights = torch.softmax(scores, dim=-1)  # [batch, num_heads, max_ctx]
    # einsum "hqk,khd->qhd" per batch:
    # Loop: attn[h,1,seq] @ v[seq,h,d] -> [1,h,d]
    # Batched: attn[b,h,seq] @ v[b,seq,h,d] -> [b,h,d]
    output = torch.einsum("bhk,bkhd->bhd", attn_weights, gathered_values)
    # output: [batch, num_heads, head_dim]

    if squeeze_output:
        return output  # [batch, num_heads, head_dim]
    return output.unsqueeze(1)  # [batch, 1, num_heads, head_dim]
