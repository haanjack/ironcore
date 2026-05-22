# Copyright (c) 2025-2026 Jaegeun Han
#
# SPDX-License-Identifier: Apache-2.0
#
# Paged attention utilities for block-based KV cache.

"""
Paged attention: gather non-contiguous physical blocks into contiguous tensors
for use with standard attention implementations.

Phase 1 approach: gather-then-attend. Physical blocks are gathered into a
contiguous tensor, which is then fed to existing Attention methods.
This avoids custom CUDA kernels while being correct and sufficiently
performant for GRPO rollout workloads.
"""

import torch


def gather_kv_blocks(
    physical_cache: torch.Tensor,
    block_table_row: torch.Tensor,
    num_valid_blocks: int,
    total_tokens: int,
    block_size: int,
) -> torch.Tensor:
    """Gather a sequence's KV blocks into a contiguous tensor.

    Args:
        physical_cache: [num_physical_blocks, block_size, num_groups, head_dim]
        block_table_row: [max_num_blocks_per_seq] logical→physical block mapping
        num_valid_blocks: Number of allocated blocks for this sequence
        total_tokens: Total number of tokens to gather (may be < num_valid_blocks * block_size
                       for a partially-filled last block)
        block_size: Tokens per block

    Returns:
        [1, total_tokens, num_groups, head_dim] contiguous KV tensor
    """
    if total_tokens == 0:
        return physical_cache.new_zeros(1, 0, physical_cache.shape[2], physical_cache.shape[3])

    num_full_blocks = total_tokens // block_size
    remainder = total_tokens % block_size

    valid_indices = block_table_row[:num_valid_blocks].long()

    if num_full_blocks == 0:
        # Only a partial first block
        idx = valid_indices[0]
        gathered = physical_cache[idx, :remainder].unsqueeze(0)
    elif remainder == 0:
        # All blocks fully filled — index by num_full_blocks, not num_valid_blocks,
        # to avoid over-gathering pre-allocated-but-not-yet-written blocks.
        full_indices = block_table_row[:num_full_blocks].long()
        gathered = physical_cache[full_indices].reshape(
            1, total_tokens, -1, physical_cache.shape[3]
        )
    else:
        # Full blocks + partial last block
        full_indices = valid_indices[:num_full_blocks]
        last_idx = valid_indices[num_full_blocks]

        full_blocks = physical_cache[full_indices].reshape(
            -1, physical_cache.shape[2], physical_cache.shape[3]
        )
        partial_block = physical_cache[last_idx, :remainder]

        gathered = torch.cat([full_blocks, partial_block], dim=0).unsqueeze(0)

    return gathered


def gather_kv_blocks_batched(
    physical_cache: torch.Tensor,
    block_tables: torch.Tensor,
    seq_ids: list[int],
    num_valid_blocks: list[int],
    token_positions: list[int],
    block_size: int,
) -> torch.Tensor:
    """Gather KV blocks for multiple sequences into a padded batched tensor.

    Vectorized alternative to calling gather_kv_blocks in a Python loop.
    Uses advanced indexing to gather all sequences' blocks simultaneously.

    Args:
        physical_cache: [num_physical_blocks, block_size, num_groups, head_dim]
        block_tables: [max_batch_size, max_num_blocks_per_seq] logical→physical mapping
        seq_ids: List of sequence IDs to gather
        num_valid_blocks: Number of allocated blocks per sequence (len == len(seq_ids))
        token_positions: Total tokens written per sequence (len == len(seq_ids))
        block_size: Tokens per block

    Returns:
        [batch, max_seq_len, num_groups, head_dim] padded contiguous KV tensor
    """
    if len(seq_ids) == 0:
        return physical_cache.new_zeros(0, 0, physical_cache.shape[2], physical_cache.shape[3])

    batch_size = len(seq_ids)
    ng = physical_cache.shape[2]
    hd = physical_cache.shape[3]
    device = physical_cache.device
    dtype = physical_cache.dtype

    max_len = max(token_positions) if token_positions else 0

    if max_len == 0:
        return physical_cache.new_zeros(batch_size, 0, ng, hd)

    # Collect (num_full_blocks, remainder) per sequence and gather all full block
    # indices into one flat list for a single advanced-indexing operation.
    per_seq_full_count: list[int] = []
    per_seq_remainder: list[int] = []
    flat_tensors: list[torch.Tensor] = []

    for i, sid in enumerate(seq_ids):
        total = token_positions[i]
        num_full = total // block_size
        remainder = total % block_size
        per_seq_full_count.append(num_full)
        per_seq_remainder.append(remainder)
        if num_full > 0:
            flat_tensors.append(block_tables[sid, :num_full].long())

    # Single gather for all full blocks — stays on-device, no Python list round-trip
    if flat_tensors:
        flat_idx = torch.cat(flat_tensors, dim=0)
        gathered_flat = physical_cache[flat_idx]  # [total_full_blocks, block_size, ng, hd]
    else:
        gathered_flat = physical_cache.new_zeros(0, block_size, ng, hd)

    # Build padded output
    result = torch.zeros(batch_size, max_len, ng, hd, device=device, dtype=dtype)
    flat_offset = 0

    for i in range(batch_size):
        sid = seq_ids[i]
        num_full = per_seq_full_count[i]
        remainder = per_seq_remainder[i]

        # Write full blocks from flat gather
        if num_full > 0:
            blocks = gathered_flat[flat_offset : flat_offset + num_full]
            tokens = num_full * block_size
            result[i, :tokens] = blocks.reshape(tokens, ng, hd)
            flat_offset += num_full

        # Write partial last block
        if remainder > 0:
            partial_block_idx = block_tables[sid, num_full].item()
            result[i, num_full * block_size : num_full * block_size + remainder] = physical_cache[
                partial_block_idx, :remainder
            ]

    return result
