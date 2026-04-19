# Copyright (c) 2025-2026 Jaegeun Han
#
# SPDX-License-Identifier: Apache-2.0
"""
Pinned (page-locked) host memory pool for weight streaming.

Allocates host memory in fixed-size chunks via cudaMallocHost (torch.cuda.pin_memory).
Page-locked memory enables async DMA transfers on CUDA streams without staging
through pageable buffers.

M2 scope: weight streaming for training. The pool is shared across all offload
subsystems (weight streaming, activation spilling in M3, backward prefetch in M4).

Allocation failure is a hard error at startup. No fallback to pageable memory.
"""

from __future__ import annotations

import threading
from collections.abc import Iterator
from contextlib import contextmanager
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from ironcore.offload.config import OffloadConfig


class PinnedMemoryPool:
    """
    Pool of page-locked host memory for DMA transfers.

    Memory is allocated in fixed-size chunks on first demand. Each allocation
    carves a slice from a chunk. When a chunk is exhausted, a new one is
    allocated. Freed allocations return to a free list within their chunk.

    Thread-safe: allocate() and free() are internally synchronized.

    Args:
        chunk_bytes: Size of each pinned chunk in bytes (default: 4GB)
        max_total_bytes: Optional cap on total pool size. Allocation fails
            with a hard error if exceeded.
    """

    def __init__(
        self,
        chunk_bytes: int = 4 * 1024 * 1024 * 1024,
        max_total_bytes: int | None = None,
    ):
        self._chunk_bytes = chunk_bytes
        self._max_total_bytes = max_total_bytes
        self._chunks: list[_PinnedChunk] = []
        self._total_allocated = 0
        self._total_used = 0
        self._lock = threading.Lock()

    @classmethod
    def from_config(cls, config: OffloadConfig) -> PinnedMemoryPool:
        """Create a pool from OffloadConfig."""
        chunk_bytes = int(config.pinned_chunk_gb * 1024**3)
        max_bytes = (
            int(config.pinned_memory_pool_gb * 1024**3)
            if config.pinned_memory_pool_gb > 0
            else None
        )
        return cls(chunk_bytes=chunk_bytes, max_total_bytes=max_bytes)

    def allocate(self, numel: int, dtype: torch.dtype) -> torch.Tensor:
        """
        Allocate a pinned host tensor of the given size and dtype.

        Args:
            numel: Number of elements
            dtype: Tensor dtype (e.g. torch.float32)

        Returns:
            Pinned host tensor of shape (numel,) on CPU

        Raises:
            RuntimeError: If allocation fails (out of host memory or budget exceeded)
        """
        element_bytes = torch.tensor([], dtype=dtype).element_size()
        requested_bytes = numel * element_bytes

        with self._lock:
            # Check budget
            if self._max_total_bytes is not None:
                if self._total_used + requested_bytes > self._max_total_bytes:
                    raise RuntimeError(
                        f"PinnedMemoryPool budget exceeded: "
                        f"{self._total_used / 1024**3:.1f}GB used + "
                        f"{requested_bytes / 1024**3:.1f}GB requested > "
                        f"{self._max_total_bytes / 1024**3:.1f}GB limit. "
                        f"Increase offload.pinned_memory_pool_gb in your config."
                    )

            # Try to fit in existing chunks
            for chunk in self._chunks:
                tensor = chunk.try_allocate(numel, dtype)
                if tensor is not None:
                    self._total_used += requested_bytes
                    return tensor

            # Need a new chunk
            if requested_bytes > self._chunk_bytes:
                # Oversized allocation: dedicated chunk
                chunk = _PinnedChunk.allocate(requested_bytes)
            else:
                chunk = _PinnedChunk.allocate(self._chunk_bytes)

            self._chunks.append(chunk)
            self._total_allocated += chunk.capacity_bytes

            tensor = chunk.try_allocate(numel, dtype)
            if tensor is None:
                # Should never happen for a fresh chunk
                raise RuntimeError(
                    f"Failed to allocate {numel} elements from fresh chunk "
                    f"(chunk capacity: {chunk.capacity_bytes} bytes)"
                )

            self._total_used += requested_bytes
            return tensor

    def free(self, tensor: torch.Tensor) -> None:
        """
        Return a previously allocated tensor to the pool.

        The memory is not returned to the OS. It stays pinned and available
        for reuse within the same chunk.
        """
        with self._lock:
            element_bytes = tensor.element_size()
            freed_bytes = tensor.numel() * element_bytes

            for chunk in self._chunks:
                if chunk.try_free(tensor):
                    self._total_used -= freed_bytes
                    return

    @contextmanager
    def allocate_temp(self, numel: int, dtype: torch.dtype) -> Iterator[torch.Tensor]:
        """Context manager: allocate, yield, free."""
        tensor = self.allocate(numel, dtype)
        try:
            yield tensor
        finally:
            self.free(tensor)

    @property
    def total_allocated_bytes(self) -> int:
        """Total pinned memory allocated from the OS."""
        return self._total_allocated

    @property
    def total_used_bytes(self) -> int:
        """Total pinned memory currently in use (not freed)."""
        return self._total_used

    @property
    def utilization(self) -> float:
        """Fraction of allocated memory currently in use."""
        if self._total_allocated == 0:
            return 0.0
        return self._total_used / self._total_allocated

    def __repr__(self) -> str:
        return (
            f"PinnedMemoryPool("
            f"chunks={len(self._chunks)}, "
            f"allocated={self._total_allocated / 1024**3:.1f}GB, "
            f"used={self._total_used / 1024**3:.1f}GB, "
            f"util={self.utilization:.1%})"
        )


class _PinnedChunk:
    """
    A single contiguous block of pinned host memory.

    Manages allocations via a simple free list. Not thread-safe.
    """

    def __init__(self, storage: torch.Tensor):
        self._storage = storage  # 1D pinned tensor
        self._capacity_bytes = storage.numel() * storage.element_size()
        self._free_list: list[tuple[int, int]] = [(0, storage.numel())]
        self._live_allocations: dict[int, tuple[int, int]] = {}  # data_ptr -> (offset, numel)

    @classmethod
    def allocate(cls, num_bytes: int) -> _PinnedChunk:
        """Allocate a new pinned chunk of the given byte size."""
        # Use uint8 for raw byte-level allocation
        numel = num_bytes  # uint8: 1 byte per element
        try:
            storage = torch.empty(numel, dtype=torch.uint8, pin_memory=True)
        except torch.cuda.OutOfMemoryError as e:
            raise RuntimeError(
                f"Failed to pin {num_bytes / 1024**3:.1f}GB of host memory. "
                f"Reduce offload.pinned_chunk_gb or free host RAM."
            ) from e
        return cls(storage)

    @property
    def capacity_bytes(self) -> int:
        return self._capacity_bytes

    def try_allocate(self, numel: int, dtype: torch.dtype) -> torch.Tensor | None:
        """
        Try to allocate a tensor from this chunk.

        Returns None if no contiguous free region is large enough.
        """
        element_bytes = torch.tensor([], dtype=dtype).element_size()
        required_bytes = numel * element_bytes

        # Search free list for a region that fits
        for i, (offset, free_numel) in enumerate(self._free_list):
            free_bytes = free_numel  # uint8 storage: numel == bytes
            if free_bytes >= required_bytes:
                # Found space
                used_bytes = required_bytes
                remaining_bytes = free_bytes - used_bytes

                # Create a view into the storage at the correct offset
                # Cast the uint8 region to the requested dtype
                byte_offset = offset
                element_size = element_bytes

                # Verify alignment
                if byte_offset % element_size != 0:
                    # Misaligned. Try next free region.
                    # This is rare since we always allocate from aligned offsets.
                    continue

                byte_slice = self._storage.narrow(0, byte_offset, required_bytes)
                tensor = byte_slice.view(dtype)

                # Update free list
                self._free_list.pop(i)
                if remaining_bytes > 0:
                    self._free_list.insert(i, (offset + used_bytes, remaining_bytes))

                # Track allocation
                self._live_allocations[tensor.data_ptr()] = (offset, used_bytes)
                return tensor

        return None

    def try_free(self, tensor: torch.Tensor) -> bool:
        """
        Try to free a tensor from this chunk.

        Returns True if the tensor belongs to this chunk and was freed.
        """
        ptr = tensor.data_ptr()
        if ptr not in self._live_allocations:
            return False

        offset, num_bytes = self._live_allocations.pop(ptr)

        # Coalesce with adjacent free regions
        merged_start = offset
        merged_end = offset + num_bytes

        new_free_list = []
        for region_start, region_numel in self._free_list:
            region_end = region_start + region_numel

            if region_end == merged_start:
                # Adjacent before: extend
                merged_start = region_start
            elif region_start == merged_end:
                # Adjacent after: extend
                merged_end = region_end
            else:
                new_free_list.append((region_start, region_numel))

        new_free_list.append((merged_start, merged_end - merged_start))
        self._free_list = new_free_list
        return True
