# Copyright (c) 2025-2026 Jaegeun Han
#
# SPDX-License-Identifier: Apache-2.0
"""
RAM-First Staircase Scaling: offload optimizer states and weights to host RAM.

Opt-in via YAML config. No changes to existing training behavior unless enabled.

Optimizer state offloading (pageable host memory, sync transfers)
Weight streaming (pinned host memory, async DMA, scheduler)

Note: weight streaming classes (PinnedMemoryPool, ExecutionScheduler, TileManager,
MemoryTransferEngine) are not imported here to avoid circular imports.
Import them directly from their modules where needed.
"""

from ironcore.config import OffloadConfig

__all__ = ["OffloadConfig"]
