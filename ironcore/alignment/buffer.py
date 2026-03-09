# Copyright (c) 2025-2026 Jaegeun Han
#
# SPDX-License-Identifier: Apache-2.0

"""Rollout buffer for GRPO with hybrid memory management.

This module provides the RolloutBuffer dataclass for storing generated
completions during GRPO training. Supports:
- GPU storage (default, fastest)
- CPU offload with pinned memory (for memory-constrained cases)
- File serialization (for checkpointing and analysis)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    pass


@dataclass
class RolloutBuffer:
    """Storage for rollout data during a training step.

    Default: GPU storage for fastest access during loss computation.
    Optional: CPU offload via .to("cpu").pin_memory() for memory-constrained cases.
    Serialization: .save()/.load() for checkpointing and offline analysis.

    Memory footprint (typical B=16, G=4, len=1536): ~1MB per buffer.
    This is negligible compared to model weights (~14GB for 7B bf16).

    Attributes:
        prompt_ids: Original prompts [B, prompt_len]
        prompt_attention_mask: Attention mask for prompts [B, prompt_len]
        completion_ids: Full sequences (prompt + response) [B*G, total_len]
        response_ids: Generated responses only [B*G, gen_len]
        old_log_probs: Log probs from generation [B*G]
        rewards: Rewards for each completion [B*G]
        advantages: Normalized advantages [B*G]
        group_ids: Group assignment for each completion [B*G]
        metadata: Metadata for reward computation [B*G]
    """

    # Prompt data (original batch)
    prompt_ids: torch.Tensor  # [B, prompt_len]
    prompt_attention_mask: torch.Tensor  # [B, prompt_len]

    # Completion data (expanded: B*G)
    completion_ids: torch.Tensor  # [B*G, total_len]
    response_ids: torch.Tensor  # [B*G, gen_len]

    # Log probabilities from generation
    old_log_probs: torch.Tensor  # [B*G]

    # Rewards and advantages (populated after reward computation)
    rewards: torch.Tensor  # [B*G]
    advantages: torch.Tensor  # [B*G]

    # Group assignment
    group_ids: torch.Tensor  # [B*G]

    # Metadata for reward computation
    metadata: list[dict]  # [B*G]

    # Optional tracking
    step: int = 0
    generation_config: dict = field(default_factory=dict)

    @property
    def batch_size(self) -> int:
        return self.prompt_ids.size(0)

    @property
    def group_size(self) -> int:
        return len(self.metadata) // self.batch_size

    @property
    def total_samples(self) -> int:
        return self.completion_ids.size(0)

    @property
    def prompt_length(self) -> int:
        return self.prompt_ids.size(1)

    @property
    def response_length(self) -> int:
        return self.response_ids.size(1)

    def to(self, device: torch.device | str) -> RolloutBuffer:
        """Move all tensors to specified device."""
        return RolloutBuffer(
            prompt_ids=self.prompt_ids.to(device),
            prompt_attention_mask=self.prompt_attention_mask.to(device),
            completion_ids=self.completion_ids.to(device),
            response_ids=self.response_ids.to(device),
            old_log_probs=self.old_log_probs.to(device),
            rewards=self.rewards.to(device),
            advantages=self.advantages.to(device),
            group_ids=self.group_ids.to(device),
            metadata=self.metadata,
            step=self.step,
            generation_config=self.generation_config,
        )

    def pin_memory(self) -> RolloutBuffer:
        """Pin memory for async GPU transfer. Call after .to('cpu')."""
        return RolloutBuffer(
            prompt_ids=self.prompt_ids.pin_memory(),
            prompt_attention_mask=self.prompt_attention_mask.pin_memory(),
            completion_ids=self.completion_ids.pin_memory(),
            response_ids=self.response_ids.pin_memory(),
            old_log_probs=self.old_log_probs.pin_memory(),
            rewards=self.rewards.pin_memory(),
            advantages=self.advantages.pin_memory(),
            group_ids=self.group_ids.pin_memory(),
            metadata=self.metadata,
            step=self.step,
            generation_config=self.generation_config,
        )

    def save(self, path: str | Path) -> None:
        """Save buffer to disk for checkpointing or analysis."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Save tensors
        torch.save(
            {
                "prompt_ids": self.prompt_ids.cpu(),
                "prompt_attention_mask": self.prompt_attention_mask.cpu(),
                "completion_ids": self.completion_ids.cpu(),
                "response_ids": self.response_ids.cpu(),
                "old_log_probs": self.old_log_probs.cpu(),
                "rewards": self.rewards.cpu(),
                "advantages": self.advantages.cpu(),
                "group_ids": self.group_ids.cpu(),
            },
            path / "tensors.pt",
        )

        # Save metadata
        with open(path / "metadata.json", "w") as f:
            json.dump(
                {
                    "step": self.step,
                    "generation_config": self.generation_config,
                    "metadata": self.metadata,
                },
                f,
            )

    @classmethod
    def load(cls, path: str | Path, device: torch.device | str = "cpu") -> RolloutBuffer:
        """Load buffer from disk."""
        path = Path(path)
        tensors = torch.load(path / "tensors.pt", map_location=device)

        with open(path / "metadata.json") as f:
            meta = json.load(f)

        return cls(
            prompt_ids=tensors["prompt_ids"],
            prompt_attention_mask=tensors["prompt_attention_mask"],
            completion_ids=tensors["completion_ids"],
            response_ids=tensors["response_ids"],
            old_log_probs=tensors["old_log_probs"],
            rewards=tensors["rewards"],
            advantages=tensors["advantages"],
            group_ids=tensors["group_ids"],
            metadata=meta["metadata"],
            step=meta["step"],
            generation_config=meta["generation_config"],
        )

    def get_group(self, group_idx: int) -> dict:
        """Get all completions for a specific prompt group."""
        mask = self.group_ids == group_idx
        indices = mask.nonzero(as_tuple=True)[0].tolist()
        return {
            "completion_ids": self.completion_ids[mask],
            "response_ids": self.response_ids[mask],
            "old_log_probs": self.old_log_probs[mask],
            "rewards": self.rewards[mask],
            "advantages": self.advantages[mask],
            "metadata": [self.metadata[i] for i in indices],
        }

    def get_best_completion(self, group_idx: int) -> dict:
        """Get the highest-reward completion for a prompt group."""
        group = self.get_group(group_idx)
        best_idx = group["rewards"].argmax().item()
        return {
            "completion_ids": group["completion_ids"][best_idx],
            "response_ids": group["response_ids"][best_idx],
            "reward": group["rewards"][best_idx].item(),
            "advantage": group["advantages"][best_idx].item(),
            "metadata": group["metadata"][best_idx],
        }

    def summary(self) -> dict:
        """Get summary statistics for logging."""
        return {
            "batch_size": self.batch_size,
            "group_size": self.group_size,
            "total_samples": self.total_samples,
            "prompt_length": self.prompt_length,
            "response_length": self.response_length,
            "mean_reward": self.rewards.mean().item(),
            "std_reward": self.rewards.std().item() if self.total_samples > 1 else 0.0,
            "mean_advantage": self.advantages.mean().item(),
            "mean_log_prob": self.old_log_probs.mean().item(),
            "step": self.step,
        }
