# Copyright (c) 2025-2026 Jaegeun Han
#
# SPDX-License-Identifier: MIT
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the above copyright notice,
# this list of conditions, and the following disclaimer are retained.
#
# Full license text is available at LICENSE file.

from typing import Literal

import torch
from torch import nn


class ActionNormalizer(nn.Module):
    """Normalizer for continuous actions.

    Supports various normalization strategies for stabilizing training.
    """

    def __init__(
        self,
        action_dim: int,
        mode: Literal["minmax", "standard", "gaussian"] = "gaussian",
        eps: float = 1e-8,
    ):
        super().__init__()

        self.action_dim = action_dim
        self.mode = mode
        self.eps = eps

        # Register buffers for statistics
        self.register_buffer("mean", torch.zeros(action_dim))
        self.register_buffer("std", torch.ones(action_dim))
        self.register_buffer("min", torch.full((action_dim,), float("inf")))
        self.register_buffer("max", torch.full((action_dim,), float("-inf")))
        self.register_buffer("initialized", torch.tensor(False))

    def fit(self, actions: torch.Tensor) -> None:
        """Compute normalization statistics from data.

        Args:
            actions: [N, action_dim] tensor of actions
        """
        if self.mode in {"gaussian", "standard"}:
            self.mean = actions.mean(dim=0)
            self.std = actions.std(dim=0).clamp(min=self.eps)

        elif self.mode == "minmax":
            self.min = actions.min(dim=0).values
            self.max = actions.max(dim=0).values
            # Handle constant dimensions
            diff = self.max - self.min
            self.min = torch.where(diff < self.eps, self.min - 1.0, self.min)
            self.max = torch.where(diff < self.eps, self.max + 1.0, self.max)

        self.initialized = torch.tensor(True)

    def forward(self, actions: torch.Tensor) -> torch.Tensor:
        """Normalize actions.

        Args:
            actions: [batch, action_dim] or [batch, horizon, action_dim]

        Returns:
            Normalized actions
        """
        if not self.initialized:
            return actions

        original_shape = actions.shape
        if actions.dim() == 3:
            actions = actions.reshape(-1, original_shape[-1])

        if self.mode in {"gaussian", "standard"}:
            normalized = (actions - self.mean) / self.std
        elif self.mode == "minmax":
            normalized = 2 * (actions - self.min) / (self.max - self.min + self.eps) - 1
        else:
            normalized = actions

        return normalized.reshape(original_shape)

    def inverse(self, normalized_actions: torch.Tensor) -> torch.Tensor:
        """Denormalize actions back to original scale.

        Args:
            normalized_actions: [batch, action_dim] or [batch, horizon, action_dim]

        Returns:
            Actions in original scale
        """
        if not self.initialized:
            return normalized_actions

        original_shape = normalized_actions.shape
        if normalized_actions.dim() == 3:
            normalized_actions = normalized_actions.reshape(-1, original_shape[-1])

        if self.mode in {"gaussian", "standard"}:
            actions = normalized_actions * self.std + self.mean
        elif self.mode == "minmax":
            actions = (normalized_actions + 1) / 2 * (self.max - self.min) + self.min
        else:
            actions = normalized_actions

        return actions.reshape(original_shape)

    def to_dict(self) -> dict:
        """Export normalizer state as dictionary."""
        return {
            "action_dim": self.action_dim,
            "mode": self.mode,
            "eps": self.eps,
            "mean": self.mean.tolist(),
            "std": self.std.tolist(),
            "min": self.min.tolist(),
            "max": self.max.tolist(),
            "initialized": self.initialized.item(),
        }

    @classmethod
    def from_dict(cls, state: dict) -> "ActionNormalizer":
        """Create normalizer from state dictionary."""
        normalizer = cls(
            action_dim=state["action_dim"],
            mode=state["mode"],
            eps=state["eps"],
        )
        normalizer.mean = torch.tensor(state["mean"])
        normalizer.std = torch.tensor(state["std"])
        normalizer.min = torch.tensor(state["min"])
        normalizer.max = torch.tensor(state["max"])
        normalizer.initialized = torch.tensor(state["initialized"])
        return normalizer
