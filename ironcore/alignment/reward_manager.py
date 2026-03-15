# Copyright (c) 2025-2026 Jaegeun Han
#
# SPDX-License-Identifier: Apache-2.0

"""RewardManager — weighted registry/orchestrator for reward functions.

Drop-in replacement for CompositeRewardFunction + get_reward_function().
Extends RewardFunction so RewardWorkerPool needs zero changes.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from .reward_model import RewardModelFunction
from .reward_rules import TemplateRuleReward
from .rewards import RewardFunction, get_reward_function

if TYPE_CHECKING:
    from ironcore.config.config_alignment import RewardConfig, RewardManagerConfig

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
        """Build RewardManager from new-style RewardManagerConfig."""
        manager = cls()
        for entry in reward_cfg.functions:
            if entry.type == "rule_template":
                if not entry.rule_template:
                    raise ValueError(f"Reward function '{entry.name}' has type='rule_template' but no rule_template path")
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
            else:
                # Delegate to existing factory for math, code, api, etc.
                kwargs = {}
                if entry.type in ("keyword", "soft_keyword"):
                    kwargs["keyword"] = entry.keyword
                if entry.type == "api":
                    kwargs["provider"] = entry.api_provider
                    if entry.api_model:
                        kwargs["model"] = entry.api_model
                if entry.type == "local_endpoint":
                    kwargs["endpoint"] = entry.local_endpoint
                if entry.type == "local_inference":
                    if entry.local_model_path:
                        kwargs["model_path"] = entry.local_model_path
                    kwargs["device"] = entry.local_device
                    kwargs["dtype"] = entry.local_dtype
                if entry.type == "composite_math":
                    kwargs["format_weight"] = entry.format_weight
                fn = get_reward_function(entry.type, **kwargs)
            manager.register(entry.name, fn, entry.weight)
        return manager

    @classmethod
    def from_legacy_config(cls, reward_cfg: RewardConfig) -> RewardManager:
        """Backward compat: wrap get_reward_function() result in a single-entry manager."""
        manager = cls()

        kwargs = {"timeout": reward_cfg.timeout}
        if reward_cfg.type == "api":
            kwargs["provider"] = reward_cfg.api_provider
            if reward_cfg.api_model:
                kwargs["model"] = reward_cfg.api_model
            if reward_cfg.prompt_template:
                kwargs["prompt_template"] = reward_cfg.prompt_template
        elif reward_cfg.type == "local_endpoint":
            kwargs["endpoint"] = reward_cfg.local_endpoint
        elif reward_cfg.type == "local_inference":
            if reward_cfg.local_model_path:
                kwargs["model_path"] = reward_cfg.local_model_path
            kwargs["device"] = reward_cfg.local_device
            kwargs["dtype"] = reward_cfg.local_dtype
            kwargs["load_in_8bit"] = reward_cfg.load_in_8bit
            kwargs["load_in_4bit"] = reward_cfg.load_in_4bit
        elif reward_cfg.type == "composite_math":
            kwargs["format_weight"] = reward_cfg.format_weight
        elif reward_cfg.type == "format":
            if reward_cfg.required_tags:
                kwargs["required_tags"] = reward_cfg.required_tags
            kwargs["penalty"] = reward_cfg.format_penalty
        elif reward_cfg.type in ("keyword", "soft_keyword"):
            kwargs["keyword"] = reward_cfg.keyword
            kwargs["case_sensitive"] = reward_cfg.keyword_case_sensitive

        fn = get_reward_function(reward_cfg.type, **kwargs)
        manager.register(reward_cfg.type, fn, weight=1.0)
        return manager
