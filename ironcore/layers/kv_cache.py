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

from ironcore.config import MainConfig
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

        # Allocate cache for each layer
        self.key_caches = []
        self.value_caches = []

        for _ in range(num_layers):
            key_cache = torch.zeros(
                batch_size,
                self.num_local_kv_groups,
                max_seq_len,
                self.head_dim,
                device=device,
                dtype=dtype,
            )
            value_cache = torch.zeros(
                batch_size,
                self.num_local_kv_groups,
                max_seq_len,
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
            for key_cache, value_cache in zip(self.key_caches, self.value_caches, strict=False):
                key_cache.zero_()
                value_cache.zero_()
        else:
            # Reset specific sequences
            self.cache_positions[batch_indices] = 0
            for key_cache, value_cache in zip(self.key_caches, self.value_caches, strict=False):
                key_cache[batch_indices].zero_()
                value_cache[batch_indices].zero_()

    def update_layer(
        self,
        layer_idx: int,
        key: torch.Tensor,
        value: torch.Tensor,
        position: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Update cache for a specific layer and return full KV tensors.

        This method appends new KV to the cache and returns the concatenated
        [cached_kv, new_kv] tensors.

        Args:
            layer_idx: Index of the transformer layer
            key: New key tensor [batch, seq_len, num_local_kv_groups, head_dim]
            value: New value tensor [batch, seq_len, num_local_kv_groups, head_dim]
            position: Optional explicit position to write to (uses cache_positions if None)

        Returns:
            Tuple of (full_key, full_value) with cached + new KV concatenated
        """
        if not self.is_initialized:
            raise RuntimeError("Cache not initialized. Call initialize() first.")

        batch_size, seq_len, num_groups, head_dim = key.shape

        # Determine write position
        if position is None:
            start_pos = self.cache_positions[0].item()  # Assume all seqs at same position
        else:
            start_pos = position

        end_pos = start_pos + seq_len

        # Check bounds
        if end_pos > self.cache_config.max_seq_length:
            raise RuntimeError(
                f"Cache overflow: trying to write to position {end_pos}, "
                f"but max_seq_length is {self.cache_config.max_seq_length}"
            )

        # Write new KV to cache
        # Transpose key/value from [batch, seq_len, num_groups, head_dim] to [batch, num_groups, seq_len, head_dim]
        key_t = key.transpose(1, 2)  # [batch, num_groups, seq_len, head_dim]
        value_t = value.transpose(1, 2)  # [batch, num_groups, seq_len, head_dim]
        self.key_caches[layer_idx][:, :, start_pos:end_pos, :] = key_t
        self.value_caches[layer_idx][:, :, start_pos:end_pos, :] = value_t

        # Update positions to reflect the highest position written
        # This ensures the cache position always reflects the current state
        if position is None:
            self.cache_positions[:] = end_pos
        else:
            # When explicit position is used, update if the write extends the cache
            self.cache_positions[:] = max(self.cache_positions.max().item(), end_pos)

        # Return full cached KV (from start to current position)
        # Transpose back to [batch, seq_len, num_groups, head_dim] for attention layer compatibility
        full_key = self.key_caches[layer_idx][:, :, :end_pos, :].transpose(
            1, 2
        )  # [batch, seq_len, num_groups, head_dim]
        full_value = self.value_caches[layer_idx][:, :, :end_pos, :].transpose(
            1, 2
        )  # [batch, seq_len, num_groups, head_dim]

        return full_key, full_value

    def get_layer_kv(
        self,
        layer_idx: int,
        start_pos: int = 0,
        end_pos: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Retrieve cached KV for a specific layer.

        Args:
            layer_idx: Index of the transformer layer
            start_pos: Start position in cache
            end_pos: End position in cache (uses current position if None)

        Returns:
            Tuple of (key_cache, value_cache)
        """
        if not self.is_initialized:
            raise RuntimeError("Cache not initialized. Call initialize() first.")

        if end_pos is None:
            end_pos = self.cache_positions[0].item()

        key = self.key_caches[layer_idx][:, :, start_pos:end_pos, :].transpose(
            1, 2
        )  # [batch, seq_len, num_groups, head_dim]
        value = self.value_caches[layer_idx][:, :, start_pos:end_pos, :].transpose(
            1, 2
        )  # [batch, seq_len, num_groups, head_dim]

        return key, value

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

        # Calculate memory usage
        total_elements = 0
        for key_cache, value_cache in zip(self.key_caches, self.value_caches, strict=False):
            total_elements += key_cache.numel() + value_cache.numel()

        bytes_per_element = torch.finfo(self.dtype).bits // 8 if self.dtype.is_floating_point else 2
        memory_mb = (total_elements * bytes_per_element) / (1024 * 1024)

        # Calculate utilization
        current_pos = self.cache_positions.max().item()
        max_pos = self.cache_config.max_seq_length
        utilization = current_pos / max_pos if max_pos > 0 else 0.0

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
