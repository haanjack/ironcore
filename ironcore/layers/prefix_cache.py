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
Prefix cache implementation for caching common prompt prefixes.

This module implements prefix caching to accelerate repeated evaluations
with shared context. The key components are:

1. PrefixCacheManager: Manages prefix detection, storage, and retrieval
2. LRU eviction policy for managing cache capacity
3. Integration with PagedKVCache for page sharing

Prefix Detection:
- Hash first N tokens (configurable, default 32+)
- Store mapping: hash -> (input_ids, page_indices, access_stats)
- Automatically detect common prefixes across batches

Benefits:
- 2-3x speedup for RLHF rollouts with long shared prompts
- Reduced memory usage through page sharing
- Faster evaluation for repeated prompt patterns
"""

import hashlib
import time

import torch


def compute_prefix_hash(input_ids: torch.Tensor, prefix_length: int = 32) -> str:
    """Compute SHA256 hash of input_ids prefix.

    Args:
        input_ids: The prefix token IDs
        prefix_length: Number of tokens to hash

    Returns:
        Hex string hash
    """
    prefix_to_hash = input_ids[:prefix_length]
    input_bytes = prefix_to_hash.cpu().numpy().tobytes()
    return hashlib.sha256(input_bytes).hexdigest()


class PrefixEntry:
    """
    Represents a cached prefix.

    Attributes:
        input_ids: The prefix token IDs
        hash: SHA256 hash of the prefix
        page_indices: Physical page indices containing the prefix KV
        access_count: Number of times this prefix has been accessed
        last_access_time: Timestamp of last access
        num_tokens: Number of tokens in the prefix
    """

    def __init__(
        self,
        input_ids: torch.Tensor,
        page_indices: list[int],
        min_prefix_length: int = 32,
    ):
        """Initialize a prefix entry.

        Args:
            input_ids: The prefix token IDs [seq_len]
            page_indices: Physical page indices containing the prefix KV
            min_prefix_length: Minimum prefix length used for hashing
        """
        self.input_ids = input_ids
        self.page_indices = page_indices
        self.access_count = 1
        self.last_access_time = time.time()
        self.num_tokens = len(input_ids)

        # Compute hash of input_ids (only first min_prefix_length tokens)
        self.hash = compute_prefix_hash(input_ids, min_prefix_length)

    def access(self):
        """Record an access to this prefix."""
        self.access_count += 1
        self.last_access_time = time.time()


class PrefixCacheManager:
    """
    Manages prefix caching with LRU eviction.

    This class detects common prefixes in input sequences and caches
    their KV tensors to avoid recomputation. Prefixes are stored with
    their associated physical page indices for efficient retrieval.

    Attributes:
        min_prefix_length: Minimum prefix length to cache
        max_pages: Maximum number of pages to use for prefix cache
        cache: Dict mapping hash to PrefixEntry
        total_pages: Total pages currently used by cached prefixes
    """

    def __init__(
        self,
        min_prefix_length: int = 32,
        max_pages: int = 1024,
        page_size: int = 16,
        eviction_policy: str = "lru",
    ):
        """Initialize prefix cache manager.

        Args:
            min_prefix_length: Minimum prefix length to consider for caching
            max_pages: Maximum number of pages to use for prefix cache
            page_size: Number of tokens per page
            eviction_policy: Eviction policy (only "lru" supported currently)
        """
        self.min_prefix_length = min_prefix_length
        self.max_pages = max_pages
        self.page_size = page_size
        self.eviction_policy = eviction_policy

        # Cache storage: hash -> PrefixEntry
        self.cache: dict[str, PrefixEntry] = {}

        # Track total pages used
        self.total_pages = 0

        # Statistics
        self.hits = 0
        self.misses = 0

    def check_prefix(
        self,
        input_ids: torch.Tensor,
    ) -> PrefixEntry | None:
        """Check if input has a cached prefix.

        Args:
            input_ids: Input token IDs [batch, seq_len] or [seq_len]

        Returns:
            PrefixEntry if found, None otherwise
        """
        # Handle both batched and unbatched input
        if input_ids.dim() == 2:
            input_ids = input_ids[0]  # Use first sequence in batch

        # Check if prefix is long enough to consider caching
        if len(input_ids) < self.min_prefix_length:
            self.misses += 1
            return None

        # Take prefix of appropriate length (for hashing)
        prefix_ids = input_ids[: self.min_prefix_length]

        # Compute hash
        hash_val = compute_prefix_hash(prefix_ids, self.min_prefix_length)

        # Check cache
        if hash_val in self.cache:
            entry = self.cache[hash_val]
            entry.access()
            self.hits += 1
            return entry

        self.misses += 1
        return None

    def save_prefix(
        self,
        input_ids: torch.Tensor,
        page_indices: list[int],
    ) -> PrefixEntry | None:
        """Save a prefix to the cache.

        Args:
            input_ids: Input token IDs (the full prefix)
            page_indices: Physical page indices containing the prefix KV

        Returns:
            The created PrefixEntry

        Raises:
            RuntimeError: If cache is full and eviction fails
        """
        # Handle both batched and unbatched input
        if input_ids.dim() == 2:
            input_ids = input_ids[0]

        # Check if prefix is long enough
        if len(input_ids) < self.min_prefix_length:
            return None

        # Create entry with min_prefix_length
        entry = PrefixEntry(input_ids, page_indices, self.min_prefix_length)

        # Check if we need to evict
        pages_needed = len(page_indices)
        if self.total_pages + pages_needed > self.max_pages:
            self._evict(pages_needed)

        # Add to cache
        self.cache[entry.hash] = entry
        self.total_pages += pages_needed

        return entry

    def load_prefix(
        self,
        entry: PrefixEntry,
        target_sequence_id: int,
        page_table,
    ):
        """Share prefix pages with a target sequence.

        Args:
            entry: The prefix entry to load
            target_sequence_id: Target sequence ID
            page_table: PageTable to use for sharing
        """
        # Create a temporary sequence ID for the cached prefix
        # This is a bit of a hack - we use the hash as a pseudo-sequence ID
        temp_id = hash(entry.hash) % (10**9)  # Use hash mod 1e9 as ID

        # Make sure the cached prefix has pages allocated
        if temp_id not in page_table.block_tables:
            # Allocate pages for the cached prefix if not already allocated
            page_table.allocate_sequence(temp_id, len(entry.page_indices))
            # Update the block table with the cached page indices
            page_table.block_tables[temp_id] = entry.page_indices.copy()

        # Share pages with target sequence
        page_table.share_pages(
            from_sequence_id=temp_id,
            to_sequence_id=target_sequence_id,
            num_pages=len(entry.page_indices),
        )

    def _evict(self, pages_needed: int):
        """Evict prefixes using LRU policy.

        Args:
            pages_needed: Number of pages that need to be freed

        Raises:
            RuntimeError: If unable to free enough pages
        """
        if self.eviction_policy != "lru":
            raise ValueError(f"Unsupported eviction policy: {self.eviction_policy}")

        # Sort entries by last access time (oldest first)
        sorted_entries = sorted(
            self.cache.values(),
            key=lambda e: e.last_access_time,
        )

        pages_freed = 0
        to_remove = []

        for entry in sorted_entries:
            if pages_freed >= pages_needed:
                break

            to_remove.append(entry.hash)
            pages_freed += len(entry.page_indices)

        # Remove evicted entries
        for hash_val in to_remove:
            del self.cache[hash_val]

        self.total_pages -= pages_freed

        if pages_freed < pages_needed:
            raise RuntimeError(
                f"Failed to evict enough pages: needed {pages_needed}, freed {pages_freed}"
            )

    def get_statistics(self) -> dict:
        """Get cache statistics.

        Returns:
            Dictionary with cache statistics
        """
        total_accesses = self.hits + self.misses
        hit_rate = self.hits / total_accesses if total_accesses > 0 else 0.0

        return {
            "min_prefix_length": self.min_prefix_length,
            "max_pages": self.max_pages,
            "total_pages_used": self.total_pages,
            "num_cached_prefixes": len(self.cache),
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": hit_rate,
            "utilization": self.total_pages / self.max_pages if self.max_pages > 0 else 0.0,
        }

    def clear(self):
        """Clear the entire cache."""
        self.cache.clear()
        self.total_pages = 0
        self.hits = 0
        self.misses = 0
