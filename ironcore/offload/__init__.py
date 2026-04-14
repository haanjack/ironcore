# Copyright (c) 2025-2026 Jaegeun Han
#
# SPDX-License-Identifier: Apache-2.0
"""
RAM-First Staircase Scaling: offload optimizer states and weights to host RAM.

Opt-in via YAML config. No changes to existing training behavior unless enabled.
"""

from ironcore.offload.config import OffloadConfig

__all__ = ["OffloadConfig"]
