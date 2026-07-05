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

    # Per-sequence actual response lengths (in tokens, excluding padding after EOS)
    # None if not tracked (e.g., loaded from old checkpoints)
    response_lengths: torch.Tensor | None = None  # [B*G]

    # Optional tracking
    step: int = 0
    generation_config: dict = field(default_factory=dict)

    @property
    def batch_size(self) -> int:
        return self.prompt_ids.size(0)

    @property
    def group_size(self) -> int:
        """Average completions per prompt in this buffer.

        Derived from group_ids (which select() slices correctly) rather than
        len(metadata) // batch_size: select() intentionally keeps prompt_ids
        (and therefore batch_size) as the original full prompt set, so that
        ratio goes wrong for a select()-produced sub-buffer that only holds a
        subset of completions.
        """
        num_groups = self.group_ids.unique().numel()
        if num_groups == 0:
            return 0
        return self.total_samples // num_groups

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
            response_lengths=self.response_lengths.to(device)
            if self.response_lengths is not None
            else None,
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
            response_lengths=self.response_lengths.pin_memory()
            if self.response_lengths is not None
            else None,
            step=self.step,
            generation_config=self.generation_config,
        )

    def save(self, path: str | Path) -> None:
        """Save buffer to disk for checkpointing or analysis."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Save tensors
        tensors = {
            "prompt_ids": self.prompt_ids.cpu(),
            "prompt_attention_mask": self.prompt_attention_mask.cpu(),
            "completion_ids": self.completion_ids.cpu(),
            "response_ids": self.response_ids.cpu(),
            "old_log_probs": self.old_log_probs.cpu(),
            "rewards": self.rewards.cpu(),
            "advantages": self.advantages.cpu(),
            "group_ids": self.group_ids.cpu(),
        }
        if self.response_lengths is not None:
            tensors["response_lengths"] = self.response_lengths.cpu()
        torch.save(tensors, path / "tensors.pt")

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
            response_lengths=tensors.get("response_lengths"),
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

    def select(self, indices: torch.Tensor) -> RolloutBuffer:
        """Select a subset of samples from the buffer (for micro-batching)."""
        return RolloutBuffer(
            prompt_ids=self.prompt_ids,  # Keep original prompts
            prompt_attention_mask=self.prompt_attention_mask,
            completion_ids=self.completion_ids[indices],
            response_ids=self.response_ids[indices],
            old_log_probs=self.old_log_probs[indices],
            rewards=self.rewards[indices],
            advantages=self.advantages[indices],
            group_ids=self.group_ids[indices],
            metadata=[self.metadata[i].copy() for i in indices.tolist()],
            response_lengths=self.response_lengths[indices]
            if self.response_lengths is not None
            else None,
            step=self.step,
            generation_config=self.generation_config,
        )

    def cat(self, other: RolloutBuffer) -> RolloutBuffer:
        """Concatenate two buffers (same batch_size required).

        Used for rollout accumulation to support larger group sizes while
        keeping memory footprint constant per chunk.

        Args:
            other: Another RolloutBuffer with the same batch_size

        Returns:
            New RolloutBuffer with concatenated completion data
        """
        if self.batch_size != other.batch_size:
            raise ValueError(
                f"Cannot concatenate buffers with different batch sizes: "
                f"{self.batch_size} vs {other.batch_size}"
            )

        def _pad_seq(a: torch.Tensor, b: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            """Pad the shorter tensor along dim=1 with zeros so shapes match."""
            la, lb = a.size(1), b.size(1)
            if la == lb:
                return a, b
            if la < lb:
                pad = a.new_zeros(a.size(0), lb - la)
                a = torch.cat([a, pad], dim=1)
            else:
                pad = b.new_zeros(b.size(0), la - lb)
                b = torch.cat([b, pad], dim=1)
            return a, b

        completion_ids_a, completion_ids_b = _pad_seq(self.completion_ids, other.completion_ids)
        response_ids_a, response_ids_b = _pad_seq(self.response_ids, other.response_ids)

        if self.response_lengths is not None and other.response_lengths is not None:
            merged_response_lengths = torch.cat(
                [self.response_lengths, other.response_lengths], dim=0
            )
        else:
            merged_response_lengths = None

        return RolloutBuffer(
            prompt_ids=self.prompt_ids,
            prompt_attention_mask=self.prompt_attention_mask,
            completion_ids=torch.cat([completion_ids_a, completion_ids_b], dim=0),
            response_ids=torch.cat([response_ids_a, response_ids_b], dim=0),
            old_log_probs=torch.cat([self.old_log_probs, other.old_log_probs], dim=0),
            rewards=torch.cat([self.rewards, other.rewards], dim=0),
            advantages=torch.cat([self.advantages, other.advantages], dim=0),
            group_ids=torch.cat([self.group_ids, other.group_ids], dim=0),
            metadata=self.metadata + other.metadata,
            response_lengths=merged_response_lengths,
            step=self.step,
            generation_config=self.generation_config,
        )
