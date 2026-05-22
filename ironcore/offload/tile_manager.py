# Copyright (c) 2025-2026 Jaegeun Han
#
# SPDX-License-Identifier: Apache-2.0
"""
Tile manager for weight streaming.

Handles tiling, precision conversion, and reassembly of model weight tensors.
Large weights are split into tiles that fit in the pinned memory pool. Each tile
can be stored at a different precision (fp32, fp16, bf16) to reduce host memory
and PCIe bandwidth.

Weight streaming. Each TransformerLayer's parameters are registered
as a "weight group" that is loaded/evicted atomically during the forward pass.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from ironcore.config import OffloadConfig
    from ironcore.offload.gpu_staging_pool import GPUStagingPool
    from ironcore.offload.memory_pool import PinnedMemoryPool

from ironcore.offload._utils import _element_size


@dataclass
class WeightTile:
    """A single tile of a weight tensor on host (pinned) memory."""

    # Host-side pinned buffer (owned by PinnedMemoryPool)
    host_tensor: torch.Tensor
    # GPU-side staging buffer (borrowed from GPUStagingPool, None when not borrowed)
    gpu_tensor: torch.Tensor | None
    # Original dtype for dequantization
    original_dtype: torch.dtype
    # Storage dtype (may be lower precision)
    storage_dtype: torch.dtype
    # Slice info for reassembly: (start_idx, end_idx) in the flat parameter
    slice_start: int
    slice_end: int
    # Number of elements in original dtype
    numel: int
    # Handle from TransferEngine (set during transfer)
    transfer_handle: object | None = None

    @property
    def nbytes_host(self) -> int:
        return self.host_tensor.numel() * self.host_tensor.element_size()

    @property
    def nbytes_gpu(self) -> int:
        if self.gpu_tensor is not None:
            return self.gpu_tensor.numel() * self.gpu_tensor.element_size()
        return self.numel * _element_size(self.original_dtype)


@dataclass
class WeightGroup:
    """
    A group of weight tiles belonging to a single TransformerLayer.

    All tiles in a group are loaded and evicted atomically. This ensures
    all weights for a layer are available before the layer executes.
    """

    layer_idx: int
    tiles: list[WeightTile]
    # Parameter references for in-place .data copy
    param_refs: list[tuple[torch.nn.Parameter, int, int]]  # (param, flat_start, flat_end)

    @property
    def total_host_bytes(self) -> int:
        return sum(t.nbytes_host for t in self.tiles)

    @property
    def total_gpu_bytes(self) -> int:
        return sum(t.nbytes_gpu for t in self.tiles)


class TileManager:
    """
    Manages tiling and precision conversion for weight streaming.

    Each parameter is a single tile (no splitting). The tile manager
    provides the allocation, transfer, and reassembly logic.

    Args:
        pool: PinnedMemoryPool for host allocations
        device: Target CUDA device
        precision: Storage precision for weights on host (fp32, fp16, bf16)
    """

    def __init__(
        self,
        pool: PinnedMemoryPool,
        device: torch.device,
        precision: str = "fp32",
        gpu_pool: GPUStagingPool | None = None,
    ):
        self._pool = pool
        self._device = device
        self._storage_dtype = self._precision_to_dtype(precision)
        self._gpu_pool = gpu_pool
        self._groups: dict[int, WeightGroup] = {}

    @classmethod
    def from_config(
        cls,
        config: OffloadConfig,
        pool: PinnedMemoryPool,
        device: torch.device,
        gpu_pool: GPUStagingPool | None = None,
    ) -> TileManager:
        """Create a TileManager from OffloadConfig."""
        return cls(
            pool=pool,
            device=device,
            precision=config.weight_storage_precision,
            gpu_pool=gpu_pool,
        )

    @staticmethod
    def _precision_to_dtype(precision: str) -> torch.dtype:
        mapping = {
            "fp32": torch.float32,
            "fp16": torch.float16,
            "bf16": torch.bfloat16,
        }
        if precision not in mapping:
            raise ValueError(
                f"Invalid precision '{precision}'. Must be one of: {list(mapping.keys())}"
            )
        return mapping[precision]

    def register_layer(
        self,
        layer_idx: int,
        params: list[torch.nn.Parameter],
    ) -> WeightGroup:
        """
        Register a layer's parameters for weight streaming.

        Allocates pinned host memory and GPU staging buffers for each parameter.
        Copies initial weights to host memory.

        Args:
            layer_idx: Layer index in the model
            params: List of nn.Parameter objects to stream

        Returns:
            WeightGroup with allocated tiles
        """
        tiles = []
        param_refs = []

        for param in params:
            numel = param.numel()
            original_dtype = param.dtype

            # Allocate pinned host buffer in storage dtype
            storage_numel = numel  # same numel, different dtype
            host_tensor = self._pool.allocate(storage_numel, self._storage_dtype)

            # Copy initial weight to host (with precision conversion if needed)
            if self._storage_dtype == original_dtype:
                host_tensor.copy_(param.data.flatten())
            else:
                host_tensor.copy_(param.data.flatten().to(self._storage_dtype))

            # GPU staging: borrow from pool at prefetch time
            gpu_tensor = None

            tile = WeightTile(
                host_tensor=host_tensor,
                gpu_tensor=gpu_tensor,
                original_dtype=original_dtype,
                storage_dtype=self._storage_dtype,
                slice_start=0,
                slice_end=numel,
                numel=numel,
            )
            tiles.append(tile)
            param_refs.append((param, 0, numel))

        group = WeightGroup(
            layer_idx=layer_idx,
            tiles=tiles,
            param_refs=param_refs,
        )
        self._groups[layer_idx] = group
        return group

    def get_group(self, layer_idx: int) -> WeightGroup | None:
        """Get the WeightGroup for a layer, or None if not registered."""
        return self._groups.get(layer_idx)

    def borrow_gpu_buffers(self, group: WeightGroup) -> None:
        """Allocate GPU staging buffers from the pool for a layer's tiles."""
        if self._gpu_pool is None:
            return
        for tile in group.tiles:
            tile.gpu_tensor = self._gpu_pool.allocate(tile.numel, tile.original_dtype)

    def return_gpu_buffers(self, group: WeightGroup) -> None:
        """Return GPU staging buffers to the pool for a layer's tiles."""
        if self._gpu_pool is None:
            return
        for tile in group.tiles:
            if tile.gpu_tensor is not None:
                self._gpu_pool.free(tile.gpu_tensor)
                tile.gpu_tensor = None

    def apply_tiles_to_params(self, group: WeightGroup) -> None:
        """
        Apply GPU staging buffers to nn.Parameters.

        For CPU-resident params (weight streaming): replaces param.data with the GPU
        staging tensor, preserving nn.Parameter identity.
        For GPU-resident params: copies in-place into param.data.
        """
        for tile, (param, _start, _end) in zip(group.tiles, group.param_refs, strict=True):
            if tile.gpu_tensor is None:
                continue
            reshaped = tile.gpu_tensor.view(param.shape)
            if param.device.type == "cpu":
                # Swap CPU param.data for GPU staging buffer
                param.data = reshaped
            else:
                param.data.copy_(reshaped)

    def snapshot_params_to_host(self, group: WeightGroup) -> None:
        """
        Copy current GPU parameter values back to host tiles.

        Used after optimizer step to update the host-side copies before eviction.
        """
        for tile, (param, _start, _end) in zip(group.tiles, group.param_refs, strict=True):
            flat_param = param.data.flatten()
            if self._storage_dtype == param.dtype:
                tile.host_tensor.copy_(flat_param)
            else:
                tile.host_tensor.copy_(flat_param.to(self._storage_dtype))

    @property
    def num_groups(self) -> int:
        return len(self._groups)

    @property
    def total_host_bytes(self) -> int:
        return sum(g.total_host_bytes for g in self._groups.values())

    @property
    def total_gpu_staging_bytes(self) -> int:
        if self._gpu_pool is not None:
            return self._gpu_pool.total_used_bytes
        return sum(g.total_gpu_bytes for g in self._groups.values())

    def __repr__(self) -> str:
        return (
            f"TileManager("
            f"groups={self.num_groups}, "
            f"precision={self._storage_dtype}, "
            f"host={self.total_host_bytes / 1024**3:.1f}GB, "
            f"gpu_staging={self.total_gpu_staging_bytes / 1024**3:.1f}GB)"
        )
