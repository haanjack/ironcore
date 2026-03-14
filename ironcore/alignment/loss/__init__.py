# Copyright (c) 2025-2026 Jaegeun Han
#
# SPDX-License-Identifier: Apache-2.0
"""Alignment loss functions for preference optimization."""

from .dpo import dpo_loss
from .grpo import compute_advantages, grpo_loss
from .kl import kl_divergence, kl_divergence_from_logits

__all__ = [
    "dpo_loss",
    "grpo_loss",
    "compute_advantages",
    "kl_divergence",
    "kl_divergence_from_logits",
]
