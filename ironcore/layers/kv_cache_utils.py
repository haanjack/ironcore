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
Shared utilities for KV cache implementations.
"""

import torch

# =============================================================================
# GQA Expansion Helper
# =============================================================================


def expand_for_gqa(
    tensor: torch.Tensor,
    num_kv_heads: int,
    num_heads: int,
    kv_dim: int = 2,
) -> torch.Tensor:
    """Expand KV tensor for Grouped Query Attention (GQA).

    GQA uses fewer KV heads than query heads. This function expands the KV
    tensor to match the query head count using expand() which creates a view
    without copying data (memory efficient).

    Args:
        tensor: Input tensor with KV heads dimension
        num_kv_heads: Number of KV heads (source)
        num_heads: Number of query heads (target)
        kv_dim: Dimension index where KV heads are located

    Returns:
        Expanded tensor with num_heads instead of num_kv_heads

    Raises:
        ValueError: If num_heads is not divisible by num_kv_heads

    Example:
        >>> # Input: [batch, seq_len, 4_kv_heads, head_dim]
        >>> # Output: [batch, seq_len, 8_query_heads, head_dim]
        >>> expanded = expand_for_gqa(kv_tensor, num_kv_heads=4, num_heads=8)
    """
    if num_heads == num_kv_heads:
        return tensor

    if num_heads % num_kv_heads != 0:
        raise ValueError(
            f"num_heads ({num_heads}) must be divisible by num_kv_heads ({num_kv_heads})"
        )

    num_heads_per_group = num_heads // num_kv_heads

    # Get the shape up to and after kv_dim
    shape = list(tensor.shape)

    # Build expansion shape: insert num_heads_per_group after kv_dim
    # e.g., [batch, seq_len, kv_heads, head_dim] -> [batch, seq_len, kv_heads, 1, head_dim]
    expand_shape = shape[: kv_dim + 1] + [num_heads_per_group] + shape[kv_dim + 1 :]

    # Use expand to avoid memory copy
    expanded = tensor.unsqueeze(kv_dim + 1).expand(expand_shape)

    # Reshape to merge the expanded dimension
    # e.g., [batch, seq_len, kv_heads, heads_per_group, head_dim] -> [batch, seq_len, num_heads, head_dim]
    output_shape = shape[:kv_dim] + [num_heads] + shape[kv_dim + 1 :]
    return expanded.reshape(output_shape)


# =============================================================================
# Statistics Utilities (Issue #2)
# =============================================================================


def compute_memory_mb(
    tensors: list[torch.Tensor],
    dtype: torch.dtype,
) -> float:
    """Compute total memory usage in megabytes for a list of tensors.

    Args:
        tensors: List of tensors to compute memory for
        dtype: Data type (used to determine bytes per element if tensors are empty)

    Returns:
        Total memory usage in megabytes
    """
    total_elements = sum(t.numel() for t in tensors)

    # Determine bytes per element
    if dtype.is_floating_point:
        bytes_per_element = torch.finfo(dtype).bits // 8
    else:
        bytes_per_element = 2  # Default to 16-bit for non-floating types

    return (total_elements * bytes_per_element) / (1024 * 1024)


def compute_utilization(current: int | float, maximum: int | float) -> float:
    """Compute utilization ratio with safe division.

    Args:
        current: Current usage
        maximum: Maximum capacity

    Returns:
        Utilization ratio (0.0 to 1.0), or 0.0 if maximum is 0
    """
    if maximum <= 0:
        return 0.0
    return float(current) / float(maximum)


