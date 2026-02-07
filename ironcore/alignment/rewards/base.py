# Copyright (c) 2025-2026 Jaegeun Han
#
# SPDX-License-Identifier: MIT

"""Base class for reward managers."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List

import torch


class BaseReward(ABC):
    """Abstract base class for reward computation.

    Reward managers are responsible for computing reward signals
    for alignment algorithms like GRPO that require online feedback.

    Subclasses should implement:
    - compute_reward(): Compute rewards for a batch of responses
    """

    @abstractmethod
    def compute_reward(
        self,
        prompts: List[str],
        responses: List[str],
        **kwargs: Any,
    ) -> torch.Tensor:
        """Compute rewards for a batch of prompt-response pairs.

        Args:
            prompts: List of input prompts
            responses: List of generated responses
            **kwargs: Additional arguments for reward computation

        Returns:
            Tensor of reward scores [batch_size]
        """
        pass

    def batch_compute_reward(
        self,
        prompts: List[str],
        responses: List[List[str]],
        **kwargs: Any,
    ) -> torch.Tensor:
        """Compute rewards for grouped responses (e.g., GRPO).

        Args:
            prompts: List of input prompts
            responses: List of response groups, each group has G responses
            **kwargs: Additional arguments

        Returns:
            Tensor of reward scores [batch_size, group_size]
        """
        batch_rewards = []
        for prompt, response_group in zip(prompts, responses):
            rewards = self.compute_reward(
                [prompt] * len(response_group),
                response_group,
                **kwargs,
            )
            batch_rewards.append(rewards)
        return torch.stack(batch_rewards)
