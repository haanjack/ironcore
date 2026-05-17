# Copyright (c) 2025-2026 Jaegeun Han
#
# SPDX-License-Identifier: Apache-2.0

"""Alignment module for GRPO and other alignment methods.

This module provides:
- Loss functions (DPO, GRPO, KL divergence)
- Rollout generation with prefix caching
- Reward computation with multiple backends
- Dataset utilities for GRPO
- Buffer management for rollout data
"""

from .buffer import RolloutBuffer
from .dataset import GRPODataset, GRPOSample, get_grpo_data_iterator, get_grpo_dataloader
from .loss import compute_advantages, dpo_loss, grpo_loss, kl_divergence
from .rewards import (
    APIRewardFunction,
    CodeRewardFunction,
    FormatRewardFunction,
    KeywordRewardFunction,
    LocalEndpointRewardFunction,
    LocalInferenceRewardFunction,
    MathRewardFunction,
    RewardFunction,
    RewardManager,
    RewardModelFunction,
    RewardWorkerPool,
    SoftKeywordRewardFunction,
    StrictFormatRewardFunction,
    TemplateRuleReward,
)
from .rollout import (
    generate_rollouts_batched,
    generate_rollouts_paged,
    generate_rollouts_with_prefix_cache,
)

__all__ = [
    # Loss functions
    "dpo_loss",
    "grpo_loss",
    "compute_advantages",
    "kl_divergence",
    # Dataset
    "GRPODataset",
    "GRPOSample",
    "get_grpo_dataloader",
    "get_grpo_data_iterator",
    # Buffer
    "RolloutBuffer",
    # Rollout generation
    "generate_rollouts_batched",
    "generate_rollouts_paged",
    "generate_rollouts_with_prefix_cache",
    # Rewards
    "RewardFunction",
    "RewardManager",
    "RewardWorkerPool",
    "MathRewardFunction",
    "CodeRewardFunction",
    "FormatRewardFunction",
    "StrictFormatRewardFunction",
    "KeywordRewardFunction",
    "SoftKeywordRewardFunction",
    "APIRewardFunction",
    "LocalEndpointRewardFunction",
    "LocalInferenceRewardFunction",
    "RewardModelFunction",
    "TemplateRuleReward",
]
