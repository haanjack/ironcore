# Copyright (c) 2025-2026 Jaegeun Han
#
# SPDX-License-Identifier: Apache-2.0

"""RewardManager — weighted registry/orchestrator for reward functions."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from .base import RewardFunction
from .builtin import (
    FormatRewardFunction,
    KeywordRewardFunction,
    LocalEndpointRewardFunction,
    LocalInferenceRewardFunction,
    MathRewardFunction,
    SoftKeywordRewardFunction,
    StrictFormatRewardFunction,
)
from .model import RewardModelFunction
from .template import TemplateRuleReward

if TYPE_CHECKING:
    from ironcore.config.config_alignment import RewardManagerConfig

logger = logging.getLogger(__name__)


class RewardManager(RewardFunction):
    """Weighted registry of reward functions. Drop-in for RewardWorkerPool."""

    def __init__(self):
        self._functions: list[tuple[str, float, RewardFunction]] = []

    def register(self, name: str, fn: RewardFunction, weight: float = 1.0) -> None:
        """Register a reward function with a name and weight."""
        self._functions.append((name, weight, fn))
        logger.info(f"Registered reward function '{name}' with weight {weight}")

    def compute(self, prompt: str, completion: str, metadata: dict) -> float:
        """Compute weighted sum of all registered reward functions."""
        if not self._functions:
            raise RuntimeError("No reward functions registered")
        total = 0.0
        for name, weight, fn in self._functions:
            total += weight * fn.compute(prompt, completion, metadata)
        return total

    @classmethod
    def from_config(cls, reward_cfg: RewardManagerConfig) -> RewardManager:
        """Build RewardManager from RewardManagerConfig."""
        manager = cls()
        for entry in reward_cfg.functions:
            if entry.type == "rule_template":
                if not entry.rule_template:
                    raise ValueError(
                        f"Reward function '{entry.name}' has type='rule_template' but no rule_template path"
                    )
                fn = TemplateRuleReward.from_yaml(entry.rule_template)
            elif entry.type == "reward_model":
                fn = RewardModelFunction(
                    backend=entry.rm_backend,
                    local_endpoint=entry.local_endpoint,
                    api_provider=entry.api_provider,
                    api_model=entry.api_model,
                    local_model_path=entry.local_model_path,
                    local_device=entry.local_device,
                    local_dtype=entry.local_dtype,
                )
            elif entry.type == "math":
                fn = MathRewardFunction(strict=True)
            elif entry.type == "composite_math":
                # Dense reward: format compliance + correctness
                format_weight = entry.format_weight
                correctness_weight = 1.0 - format_weight
                format_fn = StrictFormatRewardFunction(
                    pattern=r"####\s*-?\d",
                    reward=1.0,
                    penalty=0.0,
                )
                math_fn = MathRewardFunction(strict=False)
                # Register as composite entry
                manager.register(f"{entry.name}_format", format_fn, weight=format_weight)
                manager.register(f"{entry.name}_correctness", math_fn, weight=correctness_weight)
                continue  # Skip the default register at the end
            elif entry.type == "keyword":
                fn = KeywordRewardFunction(
                    keyword=entry.keyword,
                    case_sensitive=False,
                )
            elif entry.type == "soft_keyword":
                fn = SoftKeywordRewardFunction(
                    keyword=entry.keyword,
                    case_sensitive=False,
                    min_score=0.0,
                )
            elif entry.type == "format":
                fn = FormatRewardFunction(
                    penalty=-0.1,
                    reward_for_present=0.0,
                )
            elif entry.type == "local_endpoint":
                fn = LocalEndpointRewardFunction(endpoint=entry.local_endpoint)
            elif entry.type == "local_inference":
                fn = LocalInferenceRewardFunction(
                    model_path=entry.local_model_path or "",
                    device=entry.local_device,
                    dtype=entry.local_dtype,
                )
            else:
                raise ValueError(f"Unknown reward type: {entry.type}")
            manager.register(entry.name, fn, entry.weight)
        return manager
