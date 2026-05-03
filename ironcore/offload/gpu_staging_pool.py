# Copyright (c) 2025-2026 Jaegeun Han
#
# SPDX-License-Identifier: Apache-2.0
"""
GPU staging buffer pool for weight streaming.

Pre-allocated fixed-size GPU memory chunks with a free-list allocator.
Buffers are borrowed for H2D transfers and returned after the weights
are copied into nn.Parameter.data. Only prefetch_ahead + 1 layers'
worth of GPU memory is needed at any time.

Mirrors PinnedMemoryPool (host) but allocates on CUDA instead of
pinned host memory.
"""

from __future__ import annotations

import threading
from collections.abc import Iterator
from contextlib import contextmanager
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from ironcore.offload.config import OffloadConfig


_ELEMENT_SIZES: dict[torch.dtype, int] = {
    torch.float32: 4,
    torch.float16: 2,
    torch.bfloat16: 2,
    torch.int64: 8,
    torch.int32: 4,
    torch.uint8: 1,
}


def _element_size(dtype: torch.dtype) -> int:
    return _ELEMENT_SIZES.get(dtype) or torch.tensor([], dtype=dtype).element_size()


class _GPUChunk:
    """
    A single contiguous block of GPU memory.

    Manages allocations via a free list with coalescing. Not thread-safe
    (locking is handled by the outer GPUStagingPool).
    """

    def __init__(self, storage: torch.Tensor):
        self._storage = storage  # 1D uint8 tensor on CUDA
        self._capacity_bytes = storage.numel()
        self._free_list: list[tuple[int, int]] = [(0, storage.numel())]
        self._live_allocations: dict[int, tuple[int, int]] = {}  # data_ptr -> (offset, size_bytes)

    @classmethod
    def allocate(cls, num_bytes: int, device: torch.device) -> _GPUChunk:
        """Allocate a new GPU chunk of the given byte size."""
        numel = num_bytes  # uint8: 1 byte per element
        try:
            storage = torch.empty(numel, dtype=torch.uint8, device=device)
        except torch.cuda.OutOfMemoryError as e:
            raise RuntimeError(
                f"Failed to allocate {num_bytes / 1024**3:.1f}GB GPU staging memory. "
                f"Reduce offload.gpu_staging_chunk_mb or free VRAM."
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
        element_bytes = _element_size(dtype)
        required_bytes = numel * element_bytes

        for i, (offset, free_numel) in enumerate(self._free_list):
            free_bytes = free_numel  # uint8 storage: numel == bytes
            if free_bytes < required_bytes:
                continue

            byte_offset = offset
            element_size = element_bytes

            # Verify alignment
            if byte_offset % element_size != 0:
                continue

            byte_slice = self._storage.narrow(0, byte_offset, required_bytes)
            tensor = byte_slice.view(dtype)

            # Update free list
            remaining_bytes = free_bytes - required_bytes
            self._free_list.pop(i)
            if remaining_bytes > 0:
                self._free_list.insert(i, (offset + required_bytes, remaining_bytes))

            # Track allocation
            self._live_allocations[tensor.data_ptr()] = (offset, required_bytes)
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

        # Coalesce with adjacent free regions (fixed-point to handle bridges)
        merged_start = offset
        merged_end = offset + num_bytes

        changed = True
        while changed:
            changed = False
            remaining = []
            for region_start, region_numel in self._free_list:
                region_end = region_start + region_numel
                if region_end == merged_start:
                    merged_start = region_start
                    changed = True
                elif region_start == merged_end:
                    merged_end = region_end
                    changed = True
                else:
                    remaining.append((region_start, region_numel))
            self._free_list = remaining

        self._free_list.append((merged_start, merged_end - merged_start))
        return True


class GPUStagingPool:
    """
    Pool of pre-allocated GPU memory for weight staging buffers.

    Memory is allocated in fixed-size chunks. Each allocation carves a
    contiguous region from a chunk. Freed allocations return to the chunk's
    free list for reuse. No per-transfer cudaMalloc/cudaFree.

    Thread-safe: allocate() and free() are internally synchronized.

    Args:
        device: Target CUDA device
        chunk_bytes: Size of each GPU chunk in bytes (default: 256MB)
        max_total_bytes: Optional cap on total GPU memory the pool can consume
    """

    def __init__(
        self,
        device: torch.device,
        chunk_bytes: int = 256 * 1024 * 1024,
        max_total_bytes: int | None = None,
    ):
        self._device = device
        self._chunk_bytes = chunk_bytes
        self._max_total_bytes = max_total_bytes
        self._chunks: list[_GPUChunk] = []
        self._total_allocated = 0
        self._total_used = 0
        self._lock = threading.Lock()

    @classmethod
    def from_config(cls, config: OffloadConfig, device: torch.device) -> GPUStagingPool:
        """Create a pool from OffloadConfig."""
        chunk_bytes = int(config.gpu_staging_chunk_mb * 1024 * 1024)
        max_bytes = (
            int(config.gpu_staging_pool_mb * 1024 * 1024)
            if config.gpu_staging_pool_mb > 0
            else None
        )
        return cls(device=device, chunk_bytes=chunk_bytes, max_total_bytes=max_bytes)

    def allocate(self, numel: int, dtype: torch.dtype) -> torch.Tensor:
        """
        Allocate a GPU tensor of shape (numel,) with the given dtype.

        Args:
            numel: Number of elements
            dtype: Tensor dtype (e.g. torch.float32)

        Returns:
            GPU tensor on the pool's device

        Raises:
            RuntimeError: If allocation fails (out of VRAM or budget exceeded)
        """
        element_bytes = _element_size(dtype)
        requested_bytes = numel * element_bytes

        with self._lock:
            # Check budget
            if self._max_total_bytes is not None:
                if self._total_used + requested_bytes > self._max_total_bytes:
                    raise RuntimeError(
                        f"GPUStagingPool budget exceeded: "
                        f"{self._total_used / 1024**2:.1f}MB used + "
                        f"{requested_bytes / 1024**2:.1f}MB requested > "
                        f"{self._max_total_bytes / 1024**2:.1f}MB limit. "
                        f"Increase offload.gpu_staging_pool_mb in your config."
                    )

            # Try to fit in existing chunks
            for chunk in self._chunks:
                tensor = chunk.try_allocate(numel, dtype)
                if tensor is not None:
                    self._total_used += requested_bytes
                    return tensor

            # Need a new chunk
            if requested_bytes > self._chunk_bytes:
                chunk = _GPUChunk.allocate(requested_bytes, self._device)
            else:
                chunk = _GPUChunk.allocate(self._chunk_bytes, self._device)

            self._chunks.append(chunk)
            self._total_allocated += chunk.capacity_bytes

            tensor = chunk.try_allocate(numel, dtype)
            if tensor is None:
                raise RuntimeError(
                    f"Failed to allocate {numel} elements from fresh GPU chunk "
                    f"(chunk capacity: {chunk.capacity_bytes} bytes)"
                )

            self._total_used += requested_bytes
            return tensor

    def free(self, tensor: torch.Tensor) -> None:
        """
        Return a previously allocated tensor to the pool.

        The memory is not returned to CUDA. It stays allocated and available
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

    def auto_size(self, layer_byte_sizes: list[int], prefetch_layers: int) -> None:
        """
        Auto-size the pool budget based on registered layer sizes.

        Computes the budget as the sum of the largest ``prefetch_layers + 1``
        consecutive layers, then sets the pool's max budget and chunk size.
        Only runs if ``max_total_bytes`` was not explicitly set.

        Args:
            layer_byte_sizes: Total bytes for each registered layer group,
                in layer order.
            prefetch_layers: Number of layers to prefetch ahead.
        """
        if self._max_total_bytes is not None:
            return  # explicitly configured, don't override

        count = prefetch_layers + 1
        if len(layer_byte_sizes) <= count:
            # Fewer layers than the window — sum everything
            budget = sum(layer_byte_sizes)
        else:
            # Sliding window: find the largest sum of `count` consecutive layers
            window_sum = sum(layer_byte_sizes[:count])
            budget = window_sum
            for i in range(count, len(layer_byte_sizes)):
                window_sum += layer_byte_sizes[i] - layer_byte_sizes[i - count]
                budget = max(budget, window_sum)

        self._max_total_bytes = budget
        self._chunk_bytes = max(self._chunk_bytes, *layer_byte_sizes)

    @property
    def total_allocated_bytes(self) -> int:
        """Total GPU memory allocated from CUDA."""
        return self._total_allocated

    @property
    def total_used_bytes(self) -> int:
        """Total GPU memory currently in use (not freed)."""
        return self._total_used

    @property
    def utilization(self) -> float:
        """Fraction of allocated memory currently in use."""
        if self._total_allocated == 0:
            return 0.0
        return self._total_used / self._total_allocated

    def __repr__(self) -> str:
        return (
            f"GPUStagingPool("
            f"chunks={len(self._chunks)}, "
            f"allocated={self._total_allocated / 1024**2:.1f}MB, "
            f"used={self._total_used / 1024**2:.1f}MB, "
            f"util={self.utilization:.1%})"
        )
