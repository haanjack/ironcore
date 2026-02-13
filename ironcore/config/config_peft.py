# Copyright (c) 2025-2026 Jaegeun Han
#
# SPDX-License-Identifier: MIT
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the above copyright notice,
# this list of conditions, and the following disclaimer are retained.
#
# Full license text is available at LICENSE file.

from dataclasses import dataclass, field
from typing import Literal

from .config import BaseConfig


@dataclass
class LoRAConfig(BaseConfig):
    """Configuration for LoRA (Low-Rank Adaptation)."""

    r: int = 8
    """LoRA rank - dimensionality of low-rank matrices"""

    alpha: float = 16.0
    """LoRA scaling parameter - controls the magnitude of LoRA updates"""

    dropout: float = 0.0
    """Dropout probability applied to LoRA activations"""

    target_modules: list[str] = field(
        default_factory=lambda: ["q_proj", "v_proj"]
    )
    """List of module names to apply LoRA to.

    Available modules:
    - "q_proj": Query projection in attention
    - "k_proj": Key projection in attention
    - "v_proj": Value projection in attention
    - "o_proj": Output projection in attention
    - "up_proj": MLP up projection
    - "gate_proj": MLP gate projection (same as up_proj in implementation)
    - "down_proj": MLP down projection
    """

    @property
    def scaling(self) -> float:
        """Calculate LoRA scaling factor (alpha / r)."""
        return self.alpha / self.r


@dataclass
class PEFTConfig(BaseConfig):
    """Configuration for Parameter-Efficient Fine-Tuning methods."""

    method: Literal["none", "lora"] = "none"
    """PEFT method to use. Currently supports 'none' (full fine-tuning) and 'lora'."""

    lora: LoRAConfig = field(default_factory=LoRAConfig)
    """LoRA-specific configuration."""
