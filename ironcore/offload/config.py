# Copyright (c) 2025-2026 Jaegeun Han
#
# SPDX-License-Identifier: Apache-2.0
"""
Offload configuration for RAM-First Staircase Scaling.

M1 (simple host optimizer states): pageable host memory, sync transfers.
M2+ fields are declared here but not yet wired.
"""

from dataclasses import dataclass

from ironcore.config.config import BaseConfig


@dataclass
class OffloadConfig(BaseConfig):
    """
    Configuration for offloading tensors to host RAM.

    All fields default to disabled. Enable via YAML:
        offload:
          enabled: true
          optimizer_offload: true

    M1 fields (active):
        enabled, optimizer_offload, optimizer_state_precision,
        optimizer_min_param_elements

    M2 fields (active):
        weight_offload, weight_storage_precision, weight_prefetch_layers,
        gpu_staging_pool_mb, gpu_staging_chunk_mb,
        pinned_memory_pool_gb, pinned_chunk_gb

    M3 fields (active, merged with former M4):
        activation_spill, activation_spill_granularity

    Activation spilling replaces activation checkpointing when enabled.
    activation_spill=true automatically disables activation_recompute.
    """

    # Master switch. All offload features are gated on this.
    enabled: bool = False

    # M1: Optimizer state offloading
    optimizer_offload: bool = False
    optimizer_state_precision: str = "fp32"  # fp32 only for M1
    optimizer_min_param_elements: int = 65536  # skip offload for tiny params

    # M2: Weight streaming
    weight_offload: bool = False
    weight_prefetch_layers: int = 2
    weight_storage_precision: str = "fp32"  # precision for streamed weights on host
    gpu_staging_pool_mb: float = 0.0  # 0 = auto-size based on prefetch_layers * max_layer_bytes
    gpu_staging_chunk_mb: float = 256.0  # chunk size for GPU staging pool

    # M3: Activation offloading (merged forward spill + backward prefetch)
    # When enabled, replaces activation checkpointing. D2H spill during forward,
    # H2D prefetch during backward, free-after-consume to limit host memory.
    # Mutually exclusive with activation_recompute (auto-disabled with warning).
    activation_spill: bool = False
    activation_spill_granularity: str = "sub_layer"  # "sub_layer" or "full_layer"

    # Shared pinned memory pool (built at M2)
    pinned_memory_pool_gb: float = 100.0
    pinned_chunk_gb: float = 4.0
