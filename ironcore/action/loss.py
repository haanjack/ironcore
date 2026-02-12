# Copyright (c) 2025-2026 Jaegeun Han
#
# SPDX-License-Identifier: MIT
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the above copyright notice,
# this list of conditions, and the following disclaimer are retained.
#
# Full license text is available at LICENSE file.

import torch
import torch.nn.functional as F
from torch import nn

from ironcore.config.config_vla import ActionConfig


class ActionLoss(nn.Module):
    """Loss functions for continuous action prediction."""

    def __init__(self, config: ActionConfig):
        super().__init__()
        self.loss_type = config.loss_type
        self.prediction_horizon = config.prediction_horizon

    def forward(
        self,
        pred_actions: torch.Tensor,
        target_actions: torch.Tensor,
        action_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute action loss.

        Args:
            pred_actions: [batch, action_dim * horizon] or [batch, horizon, action_dim]
            target_actions: Same shape as pred_actions
            action_mask: Optional mask for valid actions

        Returns:
            Scalar loss value
        """
        # Ensure correct shape
        if pred_actions.dim() == 3:
            batch, horizon, action_dim = pred_actions.shape
            pred_actions = pred_actions.reshape(batch, horizon * action_dim)
            target_actions = target_actions.reshape(batch, horizon * action_dim)

        # Compute loss based on type
        if self.loss_type == "mse":
            loss = F.mse_loss(pred_actions, target_actions, reduction="none")
        elif self.loss_type == "l1":
            loss = F.l1_loss(pred_actions, target_actions, reduction="none")
        elif self.loss_type == "smooth_l1":
            loss = F.smooth_l1_loss(pred_actions, target_actions, reduction="none")
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")

        # Apply mask if provided
        if action_mask is not None:
            loss = loss * action_mask
            return loss.sum() / action_mask.sum().clamp(min=1)

        return loss.mean()


class ActionChunkingLoss(nn.Module):
    """Loss for action chunking with temporal consistency.

    Encourages smooth action predictions across the horizon.
    """

    def __init__(self, config: ActionConfig, temporal_weight: float = 0.1):
        super().__init__()
        self.action_loss = ActionLoss(config)
        self.temporal_weight = temporal_weight

    def forward(
        self,
        pred_actions: torch.Tensor,
        target_actions: torch.Tensor,
        action_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute action loss with temporal consistency term.

        Args:
            pred_actions: [batch, horizon, action_dim]
            target_actions: [batch, horizon, action_dim]
            action_mask: Optional mask

        Returns:
            Scalar loss value
        """
        # Main action loss
        main_loss = self.action_loss(pred_actions, target_actions, action_mask)

        # Temporal consistency loss (L2 on differences)
        if pred_actions.dim() == 3 and pred_actions.size(1) > 1:
            pred_diff = pred_actions[:, 1:] - pred_actions[:, :-1]
            target_diff = target_actions[:, 1:] - target_actions[:, :-1]
            temporal_loss = F.mse_loss(pred_diff, target_diff)
            return main_loss + self.temporal_weight * temporal_loss

        return main_loss
