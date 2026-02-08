# Copyright (c) 2025-2026 Jaegeun Han
#
# SPDX-License-Identifier: MIT

"""Trainer classes for IronCore.

This module contains specialized trainers:
- Trainer: Base trainer for standard SFT
- DPOTrainer: Direct Preference Optimization (offline)
- GRPOTrainer: Group Relative Policy Optimization (online) [planned]
"""

from ironcore.trainers.trainer import Trainer
from ironcore.trainers.dpo_trainer import DPOTrainer

__all__ = ["Trainer", "DPOTrainer"]
