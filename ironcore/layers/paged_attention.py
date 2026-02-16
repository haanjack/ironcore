# Copyright (c) 2025-2026 Jaegeun Han
#
# SPDX-License-Identifier: MIT
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the above copyright notice,
# this list of conditions and and the following disclaimer are retained.
#
# Full license text is available at LICENSE file.

"""
Paged attention implementation for memory-efficient KV cache management.

This module implements vLLM-style paged attention to reduce memory fragmentation
from variable-length sequences. The key components are:

1. PageTable: Manages logical-to-physical page mapping with refcounting
2. PagedKVCache: Extends KVCacheManager with paged memory management
3. Paged attention kernel (gather-based, to be optimized with custom CUDA kernel)

Page Structure:
- Each page contains a fixed number of tokens (page_size, default 16)
- Physical pages are allocated from a shared pool
- Multiple sequences can share pages (for prefix caching)
"""

import torch

from ironcore.config import MainConfig


class PageTable:
    """
    Manages logical-to-physical page mapping with refcounting.

    This class tracks which physical pages are allocated to which sequences,
    supports page sharing (for prefix caching), and handles reference counting
    for automatic deallocation.

    Attributes:
        max_num_pages: Maximum number of physical pages
        page_size: Number of tokens per page
        block_tables: Dict mapping sequence_id to list of physical page indices
        refcounts: Tensor tracking reference count for each physical page
        free_pages: List of available physical page indices
    """

    def __init__(
        self,
        max_num_pages: int,
        page_size: int,
        device: torch.device,
    ):
        """Initialize page table.

        Args:
            max_num_pages: Maximum number of physical pages
            page_size: Number of tokens per page
            device: Device to allocate refcounts on
        """
        self.max_num_pages = max_num_pages
        self.page_size = page_size
        self.device = device

        # Track which pages are allocated to which sequences
        # sequence_id -> list of physical page indices
        self.block_tables: dict[int, list[int]] = {}

        # Reference count for each physical page
        self.refcounts = torch.zeros(max_num_pages, dtype=torch.int32, device=device)

        # Free pages (initially all pages are free)
        self.free_pages = list(range(max_num_pages))

    def allocate_sequence(
        self,
        sequence_id: int,
        num_pages: int,
    ) -> list[int]:
        """Allocate pages for a new sequence.

        Args:
            sequence_id: Unique identifier for the sequence
            num_pages: Number of pages to allocate

        Returns:
            List of allocated physical page indices

        Raises:
            RuntimeError: If not enough free pages available
        """
        if len(self.free_pages) < num_pages:
            raise RuntimeError(
                f"Not enough free pages: requested {num_pages}, available {len(self.free_pages)}"
            )

        # Allocate pages from free list
        allocated_pages = []
        for _ in range(num_pages):
            page_idx = self.free_pages.pop()
            allocated_pages.append(page_idx)
            self.refcounts[page_idx] = 1  # Initialize refcount to 1

        # Store in block table
        self.block_tables[sequence_id] = allocated_pages

        return allocated_pages

    def free_sequence(self, sequence_id: int):
        """Free all pages for a sequence.

        Args:
            sequence_id: Unique identifier for the sequence
        """
        if sequence_id not in self.block_tables:
            return  # Sequence not allocated

        # Decrement refcounts and free pages with refcount 0
        pages_to_free = []
        for page_idx in self.block_tables[sequence_id]:
            self.refcounts[page_idx] -= 1
            if self.refcounts[page_idx] == 0:
                pages_to_free.append(page_idx)

        # Return freed pages to free list
        for page_idx in pages_to_free:
            self.free_pages.append(page_idx)

        # Remove from block table
        del self.block_tables[sequence_id]

    def extend_sequence(
        self,
        sequence_id: int,
        num_additional_pages: int,
    ) -> list[int]:
        """Extend allocation for an existing sequence.

        This method adds more pages to an already-allocated sequence without
        creating a new block table entry (which would lose existing data).

        Args:
            sequence_id: Unique identifier for the sequence
            num_additional_pages: Number of additional pages to allocate

        Returns:
            Updated list of physical page indices for the sequence

        Raises:
            RuntimeError: If sequence not found or not enough free pages
        """
        if sequence_id not in self.block_tables:
            raise RuntimeError(f"Sequence {sequence_id} not found")

        if len(self.free_pages) < num_additional_pages:
            raise RuntimeError(
                f"Not enough free pages: requested {num_additional_pages}, "
                f"available {len(self.free_pages)}"
            )

        # Allocate additional pages and append to existing block table
        for _ in range(num_additional_pages):
            page_idx = self.free_pages.pop()
            self.block_tables[sequence_id].append(page_idx)
            self.refcounts[page_idx] = 1

        return self.block_tables[sequence_id]

    def get_block_table(self, sequence_id: int) -> list[int] | None:
        """Get the block table (list of physical pages) for a sequence.

        Args:
            sequence_id: Unique identifier for the sequence

        Returns:
            List of physical page indices, or None if sequence not found
        """
        return self.block_tables.get(sequence_id)

    def share_pages(
        self,
        from_sequence_id: int,
        to_sequence_id: int,
        num_pages: int,
    ):
        """Share pages between two sequences (for prefix caching).

        This increments reference counts for the shared pages without allocating
        new physical pages.

        Args:
            from_sequence_id: Source sequence ID (already has pages allocated)
            to_sequence_id: Target sequence ID (will share pages)
            num_pages: Number of pages to share

        Raises:
            RuntimeError: If source sequence doesn't exist or doesn't have enough pages
        """
        if from_sequence_id not in self.block_tables:
            raise RuntimeError(f"Source sequence {from_sequence_id} not found")

        source_pages = self.block_tables[from_sequence_id]
        if len(source_pages) < num_pages:
            raise RuntimeError(
                f"Source sequence only has {len(source_pages)} pages, "
                f"requested to share {num_pages}"
            )

        # Get pages to share
        pages_to_share = source_pages[:num_pages]

        # Increment refcounts
        for page_idx in pages_to_share:
            self.refcounts[page_idx] += 1

        # Create block table entry for target sequence
        # Note: For proper prefix caching, the target would need a copy of the list
        self.block_tables[to_sequence_id] = pages_to_share.copy()

    def get_statistics(self) -> dict:
        """Get page table statistics.

        Returns:
            Dictionary with page table statistics
        """
        total_allocated = self.max_num_pages - len(self.free_pages)
        num_sequences = len(self.block_tables)

        return {
            "max_num_pages": self.max_num_pages,
            "page_size": self.page_size,
            "total_allocated": total_allocated,
            "total_free": len(self.free_pages),
            "num_sequences": num_sequences,
            "utilization": total_allocated / self.max_num_pages if self.max_num_pages > 0 else 0.0,
        }


class PagedKVCache:
    """
    Paged KV cache extending KVCacheManager with paged memory management.

    This class manages paged KV cache for memory-efficient attention.
    It allocates physical pages from a shared pool and manages per-sequence
    block tables for logical-to-physical page mapping.

    Physical cache structure per layer:
        physical_key_cache: [total_pages, num_local_kv_groups, page_size, head_dim]
        physical_value_cache: [total_pages, num_local_kv_groups, page_size, head_dim]

    Block tables per sequence:
        block_tables: Dict mapping sequence_id -> List[physical_page_indices]
    """

    def __init__(self, config: MainConfig):
        """Initialize paged KV cache.

        Args:
            config: MainConfig containing model and cache settings
        """
        self.config = config
        self.model_config = config.model
        self.cache_config = config.model.kv_cache

        # Calculate local KV groups for this TP rank
        self.num_local_kv_groups = (
            config.model.num_attention_groups // config.trainer.tensor_model_parallel_size
        )
        self.head_dim = config.model.head_dim

        # Paged attention settings
        self.page_size = self.cache_config.page_size
        self.max_num_pages = self.cache_config.max_num_pages

        # Physical page pool (allocated per layer)
        self.physical_key_caches = []  # List of [total_pages, num_local_kv_groups, page_size, head_dim]
        self.physical_value_caches = []  # List of [total_pages, num_local_kv_groups, page_size, head_dim]

        # Page tables
        self.page_table = None  # Will be initialized per batch
        self.sequence_positions = {}  # sequence_id -> current position in tokens

        self.is_initialized = False
        self.device = None
        self.dtype = None
        self.num_layers = 0

    def initialize(
        self,
        num_layers: int,
        device: torch.device,
        dtype: torch.dtype | None = None,
    ):
        """Allocate physical page pool.

        Args:
            num_layers: Number of transformer layers
            device: Device to allocate cache on
            dtype: Data type for cache (defaults to model dtype)

        Raises:
            ValueError: If TP configuration is incompatible with KV head count
        """
        from ironcore.utils import get_model_dtype

        if dtype is None:
            dtype = get_model_dtype(self.config)

        self.device = device
        self.dtype = dtype
        self.num_layers = num_layers

        # Validate TP configuration (only if parallel_states is initialized)
        from ironcore.parallel import parallel_states

        if parallel_states.is_model_parallel_initialized():
            tp_size = parallel_states.get_tensor_model_parallel_world_size()
            global_kv_groups = self.config.model.num_attention_groups

            if global_kv_groups % tp_size != 0:
                raise ValueError(
                    f"num_attention_groups ({global_kv_groups}) must be divisible by "
                    f"tensor_model_parallel_size ({tp_size}). "
                    f"Each TP rank needs an equal number of KV groups. "
                    f"Consider adjusting your configuration."
                )

        # Allocate physical page pool for each layer
        self.physical_key_caches = []
        self.physical_value_caches = []

        for _ in range(num_layers):
            key_cache = torch.zeros(
                self.max_num_pages,
                self.num_local_kv_groups,
                self.page_size,
                self.head_dim,
                device=device,
                dtype=dtype,
            )
            value_cache = torch.zeros(
                self.max_num_pages,
                self.num_local_kv_groups,
                self.page_size,
                self.head_dim,
                device=device,
                dtype=dtype,
            )
            self.physical_key_caches.append(key_cache)
            self.physical_value_caches.append(value_cache)

        # Initialize page table
        self.page_table = PageTable(
            max_num_pages=self.max_num_pages,
            page_size=self.page_size,
            device=device,
        )

        self.is_initialized = True

    def allocate_sequence(
        self,
        sequence_id: int,
        num_tokens: int,
    ) -> int:
        """Allocate pages for a new sequence.

        Args:
            sequence_id: Unique identifier for the sequence
            num_tokens: Number of tokens to allocate space for

        Returns:
            Number of pages allocated

        Raises:
            RuntimeError: If cache not initialized or not enough free pages
        """
        if not self.is_initialized:
            raise RuntimeError("Cache not initialized. Call initialize() first.")

        # Calculate number of pages needed
        num_pages = (num_tokens + self.page_size - 1) // self.page_size

        # Allocate pages from page table
        allocated_pages = self.page_table.allocate_sequence(sequence_id, num_pages)

        # Initialize sequence position
        self.sequence_positions[sequence_id] = 0

        return len(allocated_pages)

    def free_sequence(self, sequence_id: int):
        """Free all pages for a sequence.

        Args:
            sequence_id: Unique identifier for the sequence
        """
        if self.page_table is None:
            return

        self.page_table.free_sequence(sequence_id)
        if sequence_id in self.sequence_positions:
            del self.sequence_positions[sequence_id]

    def update_sequence(
        self,
        sequence_id: int,
        layer_idx: int,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, list[int]]:
        """Update cache for a specific sequence and layer.

        Args:
            sequence_id: Unique identifier for the sequence
            layer_idx: Index of the transformer layer
            key: New key tensor [batch, seq_len, num_local_kv_groups, head_dim]
            value: New value tensor [batch, seq_len, num_local_kv_groups, head_dim]

        Returns:
            Tuple of (full_key, full_value, block_table)
            full_key: Cached + new keys [batch, total_len, num_local_kv_groups, head_dim]
            full_value: Cached + new values [batch, total_len, num_local_kv_groups, head_dim]
            block_table: List of physical page indices for this sequence

        Note:
            The returned tensors have shape [batch, seq_len, num_groups, head_dim] to match
            the attention layer's expected input format.
        """
        if not self.is_initialized:
            raise RuntimeError("Cache not initialized. Call initialize() first.")

        batch_size, seq_len, num_groups, head_dim = key.shape

        # Get current position and block table
        current_pos = self.sequence_positions.get(sequence_id, 0)
        block_table = self.page_table.get_block_table(sequence_id)

        if block_table is None:
            raise RuntimeError(
                f"Sequence {sequence_id} not allocated. Call allocate_sequence() first."
            )

        # Transpose from [batch, seq_len, num_groups, head_dim] to [batch, num_groups, seq_len, head_dim]
        key_t = key.transpose(1, 2)  # [batch, num_groups, seq_len, head_dim]
        value_t = value.transpose(1, 2)

        # Calculate pages needed for this update
        end_pos = current_pos + seq_len
        start_page = current_pos // self.page_size
        end_page = (end_pos - 1) // self.page_size  # Inclusive
        num_pages_needed = end_page - start_page + 1

        # Allocate additional pages if needed
        current_pages = len(block_table)
        if num_pages_needed > current_pages - start_page:
            additional_pages = num_pages_needed - (current_pages - start_page)
            self.page_table.extend_sequence(sequence_id, additional_pages)
            block_table = self.page_table.get_block_table(sequence_id)

        # Vectorized write: handle each page in a single operation
        for page_offset in range(num_pages_needed):
            logical_page = start_page + page_offset
            physical_page = block_table[logical_page]

            # Calculate token range for this page
            page_start_token = logical_page * self.page_size
            page_end_token = min((logical_page + 1) * self.page_size, end_pos)

            # Calculate source indices in key_t/value_t
            src_start = max(0, page_start_token - current_pos)
            src_end = min(seq_len, page_end_token - current_pos)

            # Calculate destination offset in page
            dst_offset = current_pos % self.page_size if page_offset == 0 else 0
            dst_end = dst_offset + (src_end - src_start)

            # Write to physical cache
            self.physical_key_caches[layer_idx][physical_page, :, dst_offset:dst_end, :] = key_t[
                :, :, src_start:src_end, :
            ]
            self.physical_value_caches[layer_idx][physical_page, :, dst_offset:dst_end, :] = (
                value_t[:, :, src_start:src_end, :]
            )

        # Update sequence position
        self.sequence_positions[sequence_id] = end_pos

        # Gather full KV from physical pages
        full_key = self._gather_kv_from_pages(layer_idx, block_table, end_pos)
        full_value = self._gather_kv_from_pages(layer_idx, block_table, end_pos, is_value=True)

        return full_key, full_value, block_table

    def _gather_kv_from_pages(
        self,
        layer_idx: int,
        block_table: list[int],
        seq_len: int,
        is_value: bool = False,
    ) -> torch.Tensor:
        """Gather KV from physical pages.

        Uses vectorized indexing for efficient gathering. For maximum performance
        on GPU, consider using the Triton-based gather in triton_paged_attention.py.

        Args:
            layer_idx: Index of the transformer layer
            block_table: List of physical page indices
            seq_len: Number of tokens to gather
            is_value: Whether to gather values (vs keys)

        Returns:
            Gathered KV tensor [batch, seq_len, num_local_kv_groups, head_dim]
        """
        cache = self.physical_value_caches if is_value else self.physical_key_caches
        layer_cache = cache[layer_idx]
        device = layer_cache.device

        # Calculate number of pages needed
        num_pages = (seq_len + self.page_size - 1) // self.page_size

        # Pre-allocate output tensor
        # Output shape: [num_local_kv_groups, seq_len, head_dim]
        num_groups = layer_cache.shape[1]
        gathered = torch.zeros(
            num_groups, seq_len, self.head_dim, device=device, dtype=layer_cache.dtype
        )

        # Vectorized gather using index operations
        for i in range(min(num_pages, len(block_table))):
            physical_page = block_table[i]

            # Calculate token range for this page
            start_token = i * self.page_size
            end_token = min((i + 1) * self.page_size, seq_len)
            tokens_in_page = end_token - start_token

            # Direct slice assignment (vectorized)
            gathered[:, start_token:end_token, :] = layer_cache[
                physical_page, :, :tokens_in_page, :
            ]

        # Transpose to [seq_len, num_groups, head_dim] then add batch dim
        # [1, seq_len, num_groups, head_dim]
        gathered = gathered.transpose(0, 1).unsqueeze(0)

        return gathered

    def get_block_tables(self) -> dict[int, list[int]]:
        """Get all block tables.

        Returns:
            Dict mapping sequence_id to list of physical page indices
        """
        if self.page_table is None:
            return {}
        return self.page_table.block_tables.copy()

    def get_statistics(self) -> dict:
        """Get cache statistics.

        Returns:
            Dictionary with cache statistics
        """
        if not self.is_initialized:
            return {
                "initialized": False,
            }

        page_table_stats = self.page_table.get_statistics() if self.page_table else {}

        # Calculate memory usage
        total_elements = 0
        for key_cache, value_cache in zip(
            self.physical_key_caches, self.physical_value_caches, strict=False
        ):
            total_elements += key_cache.numel() + value_cache.numel()

        bytes_per_element = torch.finfo(self.dtype).bits // 8 if self.dtype.is_floating_point else 2
        memory_mb = (total_elements * bytes_per_element) / (1024 * 1024)

        return {
            "initialized": True,
            "num_layers": self.num_layers,
            "num_local_kv_groups": self.num_local_kv_groups,
            "page_size": self.page_size,
            "max_num_pages": self.max_num_pages,
            "memory_mb": memory_mb,
            **page_table_stats,
        }
