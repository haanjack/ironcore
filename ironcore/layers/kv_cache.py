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
        positions: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Update cache for a specific layer and return full KV tensors.

        This method appends new KV to the cache and returns the concatenated
        [cached_kv, new_kv] tensors.

        Args:
            layer_idx: Index of the transformer layer
            key: New key tensor [batch, seq_len, num_local_kv_groups, head_dim]
            value: New value tensor [batch, seq_len, num_local_kv_groups, head_dim]
            position: Optional explicit position to write to for all sequences.
                      Uses cache_positions[0] if None (assumes uniform positions).
                      Mutually exclusive with `positions`.
            positions: Optional per-sequence positions tensor [batch].
                       Each sequence can have a different starting position.
                       Mutually exclusive with `position`.

        Returns:
            Tuple of (full_key, full_value) with cached + new KV concatenated

        Note:
            - For uniform position updates (batch generation): use `position` parameter
            - For per-sequence positions (continuous batching): use `positions` tensor
            - If both are None, uses cached positions (assumes all sequences at same position)

        Raises:
            RuntimeError: If cache not initialized or overflow occurs
            ValueError: If both position and positions are specified
        """
        if not self.is_initialized:
            raise RuntimeError("Cache not initialized. Call initialize() first.")

        # Validate mutually exclusive parameters
        if position is not None and positions is not None:
            raise ValueError("Cannot specify both 'position' and 'positions' parameters")

        batch_size, seq_len, num_groups, head_dim = key.shape

        # Determine write positions
        if positions is not None:
            # Per-sequence positions (continuous batching scenario)
            start_positions = positions
            end_positions = positions + seq_len

            # Check bounds for all sequences
            if end_positions.max().item() > self.cache_config.max_seq_length:
                raise RuntimeError(
                    f"Cache overflow: trying to write to position {end_positions.max().item()}, "
                    f"but max_seq_length is {self.cache_config.max_seq_length}"
                )

            # Write new KV to cache for each sequence at its respective position
            #
            # SHAPE TRANSFORMATION:
            # Input:  key/value [batch, seq_len, num_groups, head_dim]  <- Attention format
            # Cache:  key_cache [batch, num_groups, max_seq_len, head_dim] <- Storage format
            # We transpose dims 1 and 2 to convert between formats
            key_t = key.transpose(1, 2)  # [batch, num_groups, seq_len, head_dim]
            value_t = value.transpose(1, 2)  # [batch, num_groups, seq_len, head_dim]

            for b in range(batch_size):
                start_pos = start_positions[b].item()
                end_pos = end_positions[b].item()
                self.key_caches[layer_idx][b, :, start_pos:end_pos, :] = key_t[b]
                self.value_caches[layer_idx][b, :, start_pos:end_pos, :] = value_t[b]

            # Update cache positions
            self.cache_positions = end_positions

            # Return KV - note: with per-sequence positions, we return based on max position
            # This is a limitation: callers should use get_layer_kv for per-sequence retrieval
            #
            # SHAPE TRANSFORMATION:
            # Cache:  key_cache[:, :, :max_end_pos, :] [batch, num_groups, seq_len, head_dim] <- Storage format
            # Output: full_key [batch, seq_len, num_groups, head_dim]  <- Attention format
            max_end_pos = end_positions.max().item()
            full_key = self.key_caches[layer_idx][:, :, :max_end_pos, :].transpose(
                1, 2
            )  # [batch, seq_len, num_groups, head_dim]
            full_value = self.value_caches[layer_idx][:, :, :max_end_pos, :].transpose(
                1, 2
            )  # [batch, seq_len, num_groups, head_dim]

        else:
            # Uniform position for all sequences
            if position is None:
                start_pos = self.cache_positions[0].item()  # Assumes uniform position
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
            #
            # SHAPE TRANSFORMATION:
            # Input:  key/value [batch, seq_len, num_groups, head_dim]  <- Attention format
            # Cache:  key_cache [batch, num_groups, max_seq_len, head_dim] <- Storage format
            # We transpose dims 1 and 2 to convert between formats
            key_t = key.transpose(1, 2)  # [batch, num_groups, seq_len, head_dim]
            value_t = value.transpose(1, 2)  # [batch, num_groups, seq_len, head_dim]
            self.key_caches[layer_idx][:, :, start_pos:end_pos, :] = key_t
            self.value_caches[layer_idx][:, :, start_pos:end_pos, :] = value_t

            # Update all positions uniformly
            self.cache_positions[:] = end_pos

            # Return full cached KV (from start to current position)
            #
            # SHAPE TRANSFORMATION:
            # Cache:  key_cache[:, :, :end_pos, :] [batch, num_groups, seq_len, head_dim] <- Storage format
            # Output: full_key [batch, seq_len, num_groups, head_dim]  <- Attention format
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
        batch_idx: int | torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Retrieve cached KV for a specific layer.

        Args:
            layer_idx: Index of the transformer layer
            start_pos: Start position in cache
            end_pos: End position in cache (uses cache_positions[0] if None)
            batch_idx: Optional batch index or indices to retrieve.
                       If None, retrieves for all sequences in batch.

        Returns:
            Tuple of (key_cache, value_cache)
            - If batch_idx is None: [batch, seq_len, num_groups, head_dim]
            - If batch_idx is specified: [selected_batch, seq_len, num_groups, head_dim]
        """
        if not self.is_initialized:
            raise RuntimeError("Cache not initialized. Call initialize() first.")

        if end_pos is None:
            end_pos = self.cache_positions[0].item()

        if batch_idx is not None:
            # Retrieve for specific sequence(s)
            #
            # SHAPE TRANSFORMATION:
            # Cache:  key_cache[batch_idx, :, start_pos:end_pos, :] [num_groups, seq_len, head_dim] <- Storage format
            # Output: key [seq_len, num_groups, head_dim] <- After transpose(0,1)
            # Note: For single sequence, we need to add batch dim back
            key = self.key_caches[layer_idx][batch_idx, :, start_pos:end_pos, :].transpose(
                0, 1
            )  # [seq_len, num_groups, head_dim] or [selected, seq_len, num_groups, head_dim]
            value = self.value_caches[layer_idx][batch_idx, :, start_pos:end_pos, :].transpose(
                0, 1
            )  # [seq_len, num_groups, head_dim]
            if key.dim() == 3:
                # Single sequence selected, add batch dimension
                key = key.unsqueeze(0)  # [1, seq_len, num_groups, head_dim]
                value = value.unsqueeze(0)  # [1, seq_len, num_groups, head_dim]
        else:
            # Retrieve for all sequences in batch
            #
            # SHAPE TRANSFORMATION:
            # Cache:  key_cache[:, :, start_pos:end_pos, :] [batch, num_groups, seq_len, head_dim] <- Storage format
            # Output: key [batch, seq_len, num_groups, head_dim] <- After transpose(1,2)
            key = self.key_caches[layer_idx][:, :, start_pos:end_pos, :].transpose(
                1, 2
            )  # [batch, seq_len, num_groups, head_dim]
            value = self.value_caches[layer_idx][:, :, start_pos:end_pos, :].transpose(
                1, 2
            )  # [batch, seq_len, num_groups, head_dim]

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
