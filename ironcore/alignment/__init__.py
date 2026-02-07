# Copyright (c) 2025-2026 Jaegeun Han
#
# SPDX-License-Identifier: MIT

"""Alignment algorithms for IronCore.

This module contains implementations of alignment algorithms including:
- DPO (Direct Preference Optimization)
- GRPO (Group Relative Policy Optimization) [planned]

Submodules:
- loss: Alignment-specific loss functions
- rewards: Reward managers for online/offline feedback
"""

from ironcore.alignment.loss import dpo_loss

__all__ = ["dpo_loss"]
