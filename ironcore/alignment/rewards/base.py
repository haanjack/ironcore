# Copyright (c) 2025-2026 Jaegeun Han
#
# SPDX-License-Identifier: Apache-2.0

"""Base reward function and worker pool."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor

import torch

logger = logging.getLogger(__name__)


class RewardFunction(ABC):
    """Base class for reward functions."""

    @abstractmethod
    def compute(self, prompt: str, completion: str, metadata: dict) -> float:
        """Compute reward for a completion given prompt and metadata.

        Args:
            prompt: The input prompt
            completion: The model's completion
            metadata: Additional info (answer, test_cases, etc.)

        Returns:
            Reward score, typically in [0, 1] range
        """
        pass


class RewardWorkerPool:
    """Pool of worker threads for parallel reward computation.

    Uses ThreadPoolExecutor for parallelism with timeout support.
    Threads share memory, avoiding pickling issues with ProcessPoolExecutor.

    Attributes:
        reward_fn: Reward function to compute scores
        num_workers: Maximum number of parallel workers
        timeout: Seconds before returning default reward (0.5)
        default_reward: Reward returned on timeout or error
    """

    def __init__(
        self,
        reward_fn: RewardFunction,
        num_workers: int = 4,
        timeout: float = 30.0,
        default_reward: float = 0.5,
    ):
        self.reward_fn = reward_fn
        self.num_workers = num_workers
        self.timeout = timeout
        self.default_reward = default_reward
        self._executor = ThreadPoolExecutor(max_workers=num_workers)
        # Failure tracking — exposes visibility into reward computation errors
        # so a poisoned run is observable instead of silently degrading the
        # GRPO advantage signal. (Fable issue #67.)
        self._failure_count: int = 0

    def get_failure_count(self) -> int:
        """Number of reward computations that failed or timed out since start."""
        return self._failure_count

    def score_batch(
        self,
        prompts: list[str],
        completions: list[str],
        metadata_list: list[dict],
    ) -> torch.Tensor:
        """Compute rewards for a batch of completions in parallel.

        Args:
            prompts: List of prompts
            completions: List of completions (same length)
            metadata_list: List of metadata dicts

        Returns:
            Tensor of rewards [batch_size]
        """
        assert len(prompts) == len(completions) == len(metadata_list)

        # Submit all tasks to thread pool
        futures = [
            self._executor.submit(self.reward_fn.compute, p, c, m)
            for p, c, m in zip(prompts, completions, metadata_list, strict=False)
        ]

        # Collect results with global timeout
        from concurrent.futures import wait

        done, not_done = wait(futures, timeout=self.timeout)

        rewards = []
        for idx, future in enumerate(futures):
            if future in done:
                try:
                    result = future.result()
                    rewards.append(float(result))
                except (ValueError, RuntimeError, TypeError, OSError, KeyError, AttributeError) as exc:
                    # Narrow catch: only the exception types a reward backend
                    # is realistically expected to raise. Log with context so
                    # a poisoned run is observable. (Fable issue #67.)
                    preview = (prompts[idx][:40] + "…") if len(prompts[idx]) > 40 else prompts[idx]
                    logger.warning(
                        "reward compute failed for prompt='%s': %s: %s",
                        preview,
                        type(exc).__name__,
                        exc,
                    )
                    self._failure_count += 1
                    rewards.append(self.default_reward)
            else:
                # Timeout — cancel the future if possible and return default
                future.cancel()
                logger.warning(
                    "reward compute timed out after %.1fs for prompt='%s'",
                    self.timeout,
                    (prompts[idx][:40] + "…") if len(prompts[idx]) > 40 else prompts[idx],
                )
                self._failure_count += 1
                rewards.append(self.default_reward)

        return torch.tensor(rewards, dtype=torch.float32)

    def shutdown(self):
        """Shutdown the worker pool."""
        self._executor.shutdown(wait=False)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.shutdown()
        return False
