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
KV Cache Manager for autoregressive generation.

This module provides a cache manager for storing key/value tensors during
autoregressive generation to avoid redundant computation.

Supported Features:
- Tensor Parallelism (TP) aware: stores only local KV groups per rank
- Per-sequence position tracking for variable-length sequences
- Selective cache reset for continuous batching

Limitations:
- All sequences in a batch must use the same position when `position=None`
- For mixed-position scenarios, use `positions` parameter to specify per-sequence positions
"""

import torch

from ironcore.config import MainConfig
from ironcore.layers.kv_cache_utils import compute_memory_mb, compute_utilization
from ironcore.utils import get_model_dtype


class KVCacheManager:
    """
    Manages KV cache for autoregressive generation.

    This cache stores key/value tensors during generation to avoid redundant computation.
    It's TP-aware and stores only the local shard of KV groups on each rank.

    Cache structure per layer:
        key_cache: [batch, num_local_kv_groups, max_seq_len, head_dim]
        value_cache: [batch, num_local_kv_groups, max_seq_len, head_dim]
        cache_positions: [batch]  # Current fill position per sequence

    Position Tracking:
        The cache maintains per-sequence positions in `cache_positions`.
        - For uniform position updates (all sequences at same position): use `position` parameter
        - For per-sequence positions: use `positions` tensor parameter
        - For automatic tracking: omit position parameters (assumes uniform positions)
    """

    def __init__(self, config: MainConfig):
        """Initialize cache manager with model configuration.

        Args:
            config: MainConfig containing model and trainer settings
        """
        self.config = config
        self.model_config = config.model
        self.cache_config = config.model.kv_cache

        # Calculate local KV groups for this TP rank
        self.num_local_kv_groups = (
            config.model.num_attention_groups // config.trainer.tensor_model_parallel_size
        )
        self.head_dim = config.model.head_dim

        # Cache storage (initialized lazily)
        self.key_caches = []  # List of [batch, num_local_kv_groups, max_seq_len, head_dim]
        self.value_caches = []  # List of [batch, num_local_kv_groups, max_seq_len, head_dim]
        self.cache_positions = None  # [batch] - current position per sequence

        self.is_initialized = False
        self.device = None
        self.dtype = None

    def initialize(
        self,
        batch_size: int,
        num_layers: int,
        device: torch.device,
        dtype: torch.dtype | None = None,
    ):
        """Allocate cache buffers.

        Args:
            batch_size: Number of sequences in batch
            num_layers: Number of transformer layers
            device: Device to allocate cache on
            dtype: Data type for cache (defaults to model dtype)
        """
        if dtype is None:
            dtype = get_model_dtype(self.config)

        self.device = device
        self.dtype = dtype

        max_seq_len = self.cache_config.max_seq_length

        # Allocate cache for each layer: [batch, max_seq_len, num_groups, head_dim]
        # This matches Attention layout [b, s, n, d] to avoid transposes
        self.key_caches = []
        self.value_caches = []

        for _ in range(num_layers):
            key_cache = torch.zeros(
                batch_size,
                max_seq_len,
                self.num_local_kv_groups,
                self.head_dim,
                device=device,
                dtype=dtype,
            )
            value_cache = torch.zeros(
                batch_size,
                max_seq_len,
                self.num_local_kv_groups,
                self.head_dim,
                device=device,
                dtype=dtype,
            )
            self.key_caches.append(key_cache)
            self.value_caches.append(value_cache)

        # Initialize cache positions (all start at 0)
        self.cache_positions = torch.zeros(batch_size, device=device, dtype=torch.long)

        self.is_initialized = True

    def reset(self, batch_indices: torch.Tensor | None = None):
        """Clear cache for specified sequences.

        Args:
            batch_indices: Indices of sequences to reset. If None, reset all.
        """
        if not self.is_initialized:
            return

        if batch_indices is None:
            # Reset all sequences
            self.cache_positions.zero_()
            # Note: we don't strictly need to zero the cache data, just positions
        else:
            # Reset specific sequences
            self.cache_positions[batch_indices] = 0

    def update_layer(
        self,
        layer_idx: int,
        key: torch.Tensor,
        value: torch.Tensor,
        position: int | None = None,
        positions: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Update cache for a specific layer and return full KV tensors.

        Args:
            layer_idx: Index of the transformer layer
            key: New key tensor [batch, seq_len, num_local_kv_groups, head_dim]
            value: New value tensor [batch, seq_len, num_local_kv_groups, head_dim]
            position: Optional explicit position to write to for all sequences.
            positions: Optional per-sequence positions tensor [batch].

        Returns:
            Tuple of (full_key, full_value) with cached + new KV concatenated
        """
        if not self.is_initialized:
            raise RuntimeError("Cache not initialized. Call initialize() first.")

        batch_size, seq_len, num_groups, head_dim = key.shape

        # Determine write positions
        if positions is not None:
            # Per-sequence positions (continuous batching scenario)
            # Check bounds
            if (positions + seq_len).max().item() > self.cache_config.max_seq_length:
                raise RuntimeError("Cache overflow")

            # Vectorized update using scatter_
            # idx shape: [batch, seq_len, 1, 1]
            idx = positions.view(batch_size, 1, 1, 1) + torch.arange(
                seq_len, device=key.device
            ).view(1, -1, 1, 1)
            idx = idx.expand(-1, -1, self.num_local_kv_groups, self.head_dim)

            self.key_caches[layer_idx].scatter_(1, idx, key)
            self.value_caches[layer_idx].scatter_(1, idx, value)

            # Update cache positions
            self.cache_positions = positions + seq_len
            max_end_pos = self.cache_positions.max().item()
            
            full_key = self.key_caches[layer_idx][:, :max_end_pos]
            full_value = self.value_caches[layer_idx][:, :max_end_pos]

        else:
            start_pos = position if position is not None else self.cache_positions[0].item()
            end_pos = start_pos + seq_len

            if end_pos > self.cache_config.max_seq_length:
                raise RuntimeError(f"Cache overflow: {end_pos} > {self.cache_config.max_seq_length}")

            # Direct slice assignment - NO TRANSPOSES
            self.key_caches[layer_idx][:, start_pos:end_pos] = key
            self.value_caches[layer_idx][:, start_pos:end_pos] = value

            # Update all positions uniformly
            self.cache_positions[:] = end_pos

            # Return full cached KV
            full_key = self.key_caches[layer_idx][:, :end_pos]
            full_value = self.value_caches[layer_idx][:, :end_pos]

        return full_key, full_value

    def get_layer_kv(
        self,
        layer_idx: int,
        start_pos: int = 0,
        end_pos: int | None = None,
        batch_idx: int | torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Retrieve cached KV for a specific layer."""
        if not self.is_initialized:
            raise RuntimeError("Cache not initialized")

        if end_pos is None:
            end_pos = self.cache_positions.max().item()

        if batch_idx is not None:
            key = self.key_caches[layer_idx][batch_idx, start_pos:end_pos]
            value = self.value_caches[layer_idx][batch_idx, start_pos:end_pos]
        else:
            key = self.key_caches[layer_idx][:, start_pos:end_pos]
            value = self.value_caches[layer_idx][:, start_pos:end_pos]

        return key, value

    def get_sequence_position(self, batch_idx: int) -> int:
        """Get current cache position for a specific sequence.

        Args:
            batch_idx: Index of sequence in batch

        Returns:
            Current cache position for the specified sequence
        """
        if not self.is_initialized:
            return 0
        return self.cache_positions[batch_idx].item()

    def set_sequence_position(self, batch_idx: int, position: int):
        """Set cache position for a specific sequence.

        Useful for continuous batching when sequences have different lengths.

        Args:
            batch_idx: Index of sequence in batch
            position: New position value
        """
        if self.is_initialized:
            self.cache_positions[batch_idx] = position

    def get_cache_position(self, batch_idx: int = 0) -> int:
        """Get current cache fill level for a sequence.

        Args:
            batch_idx: Index of sequence in batch

        Returns:
            Current position in cache
        """
        if not self.is_initialized:
            return 0
        return self.cache_positions[batch_idx].item()

    def get_statistics(self) -> dict:
        """Get cache statistics for monitoring.

        Returns:
            Dictionary with cache statistics
        """
        if not self.is_initialized:
            return {
                "initialized": False,
                "memory_mb": 0,
                "utilization": 0.0,
            }

        # Calculate memory usage using shared helper
        all_caches = self.key_caches + self.value_caches
        memory_mb = compute_memory_mb(all_caches, self.dtype)

        # Calculate utilization using shared helper
        current_pos = self.cache_positions.max().item()
        max_pos = self.cache_config.max_seq_length
        utilization = compute_utilization(current_pos, max_pos)

        return {
            "initialized": True,
            "num_layers": len(self.key_caches),
            "batch_size": self.cache_positions.shape[0],
            "num_local_kv_groups": self.num_local_kv_groups,
            "current_position": current_pos,
            "max_seq_length": max_pos,
            "memory_mb": memory_mb,
            "utilization": utilization,
        }
