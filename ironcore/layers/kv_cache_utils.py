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
    kv_dim: int = 1,
) -> torch.Tensor:
    """Expand KV tensor for Grouped Query Attention (GQA)."""
    if num_kv_heads == num_heads:
        return tensor

    if num_heads % num_kv_heads != 0:
        raise ValueError(
            f"num_heads ({num_heads}) must be divisible by num_kv_heads ({num_kv_heads})"
        )

    num_replicas = num_heads // num_kv_heads

    if kv_dim == 1:
        # [batch, groups, seq, dim] -> [batch, groups, 1, seq, dim] -> [batch, heads, seq, dim]
        batch, groups, seq, dim = tensor.shape
        return (
            tensor.unsqueeze(2)
            .expand(-1, -1, num_replicas, -1, -1)
            .reshape(batch, num_heads, seq, dim)
        )
    elif kv_dim == 2:
        # [batch, seq, groups, dim] -> [batch, seq, groups, 1, dim] -> [batch, seq, heads, dim]
        batch, seq, groups, dim = tensor.shape
        return (
            tensor.unsqueeze(3)
            .expand(-1, -1, -1, num_replicas, -1)
            .reshape(batch, seq, num_heads, dim)
        )
    else:
        # Generic fallback
        shape = list(tensor.shape)
        expand_shape = shape[: kv_dim + 1] + [num_replicas] + shape[kv_dim + 1 :]
        expanded = tensor.unsqueeze(kv_dim + 1).expand(expand_shape)
        output_shape = shape[:kv_dim] + [num_heads] + shape[kv_dim + 1 :]
        return expanded.reshape(output_shape)


# =============================================================================
# Statistics Utilities
# =============================================================================


def compute_memory_mb(
    tensors: list[torch.Tensor],
    dtype: torch.dtype,
) -> float:
    """Compute total memory usage in megabytes for a list of tensors."""
    total_elements = sum(t.numel() for t in tensors)

    # Determine bytes per element
    if dtype.is_floating_point:
        bytes_per_element = torch.finfo(dtype).bits // 8
    else:
        bytes_per_element = 2  # Default to 16-bit for non-floating types

    return (total_elements * bytes_per_element) / (1024 * 1024)


def compute_utilization(current: int | float, maximum: int | float) -> float:
    """Compute utilization ratio with safe division."""
    if maximum <= 0:
        return 0.0
    return float(current) / float(maximum)
