# Copyright (c) 2025-2026 Jaegeun Han
#
# SPDX-License-Identifier: Apache-2.0

"""Reward functions for GRPO alignment.

This package provides:
- Base reward function class and worker pool
- Built-in reward functions (math, code, format, keyword, API, local)
- Reward manager for weighted combinations
- Reward model integration
- YAML template-driven rule rewards
"""

from .base import RewardFunction, RewardWorkerPool
from .builtin import (
    APIRewardFunction,
    CodeRewardFunction,
    FormatRewardFunction,
    KeywordRewardFunction,
    LocalEndpointRewardFunction,
    LocalInferenceRewardFunction,
    MathRewardFunction,
    SoftKeywordRewardFunction,
    StrictFormatRewardFunction,
)
from .manager import RewardManager
from .model import RewardModelFunction
from .template import TemplateRuleReward

__all__ = [
    # Base
    "RewardFunction",
    "RewardWorkerPool",
    # Built-in reward functions
    "MathRewardFunction",
    "CodeRewardFunction",
    "FormatRewardFunction",
    "StrictFormatRewardFunction",
    "KeywordRewardFunction",
    "SoftKeywordRewardFunction",
    "APIRewardFunction",
    "LocalEndpointRewardFunction",
    "LocalInferenceRewardFunction",
    # Manager
    "RewardManager",
    # Reward model
    "RewardModelFunction",
    # Template-based rewards
    "TemplateRuleReward",
]
