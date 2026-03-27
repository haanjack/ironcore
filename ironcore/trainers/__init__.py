# Copyright (c) 2025-2026 Jaegeun Han
#
# SPDX-License-Identifier: Apache-2.0

"""Trainer classes for IronCore.

This module provides specialized trainers for different training paradigms:

Base:
- BaseTrainer: Abstract base class with common infrastructure

Language Modeling:
- LanguageModelTrainer: For pretraining and supervised fine-tuning (SFT)
- PretrainTrainer: Alias for LanguageModelTrainer (pretraining emphasis)
- SFTTrainer: Alias for LanguageModelTrainer (SFT emphasis)

Alignment:
- DPOTrainer: Direct Preference Optimization (offline RL)
- GRPOTrainer: Group Relative Policy Optimization (online RL)

Example Usage:
    # Pretraining
    from ironcore.trainers import PretrainTrainer
    trainer = PretrainTrainer(config, forward_step_func, loss_fn)

    # Supervised Fine-Tuning
    from ironcore.trainers import SFTTrainer
    trainer = SFTTrainer(config, forward_step_func, loss_fn)

    # Or use the general name
    from ironcore.trainers import LanguageModelTrainer
    trainer = LanguageModelTrainer(config, forward_step_func, loss_fn)

    # Preference Optimization (offline)
    from ironcore.trainers import DPOTrainer
    trainer = DPOTrainer(config, forward_step_func, loss_fn)

    # Group Relative Policy Optimization (online)
    from ironcore.trainers import GRPOTrainer
    trainer = GRPOTrainer(config, forward_step_func, loss_fn)
"""

from .base_trainer import BaseTrainer
from .dpo_trainer import DPOTrainer
from .grpo_trainer import GRPOTrainer
from .language_model_trainer import (
    LanguageModelTrainer,
    PretrainTrainer,
    SFTTrainer,
)

__all__ = [
    # Base
    "BaseTrainer",
    # Language Modeling
    "LanguageModelTrainer",
    "PretrainTrainer",
    "SFTTrainer",
    # Alignment
    "DPOTrainer",
    "GRPOTrainer",
]
