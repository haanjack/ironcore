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
from torch import nn

from ironcore.config import MainConfig
from ironcore.layers.module import BaseModule
from ironcore.parallel.tensor_parallel import ColumnParallelLinear, RowParallelLinear


class ActionHead(BaseModule):
    """Action prediction head for VLA models.

    Predicts continuous actions from language model hidden states.
    Uses IronCore's parallel linear layers for tensor parallelism.
    """

    def __init__(self, config: MainConfig):
        super().__init__(config)

        action_config = config.vla.action
        model_config = config.model

        self.hidden_size = model_config.d_model
        self.action_hidden_size = action_config.hidden_size
        self.action_dim = action_config.action_dim
        self.prediction_horizon = action_config.prediction_horizon
        self.num_layers = action_config.num_layers

        # Total output dimension
        self.output_dim = self.action_dim * self.prediction_horizon

        # Build MLP layers
        layers = []

        # Input layer: hidden_size -> action_hidden_size
        layers.append(
            ColumnParallelLinear(
                config,
                self.hidden_size,
                self.action_hidden_size,
                bias=True,
                gather_output=False,
            )
        )
        layers.append(nn.GELU())

        # Hidden layers
        for _ in range(self.num_layers - 2):
            layers.append(
                ColumnParallelLinear(
                    config,
                    self.action_hidden_size,
                    self.action_hidden_size,
                    bias=True,
                    gather_output=False,
                )
            )
            layers.append(nn.GELU())

        # Output layer: action_hidden_size -> output_dim
        # Use RowParallelLinear for final layer to reduce across TP ranks
        layers.append(
            RowParallelLinear(
                config,
                self.action_hidden_size,
                self.output_dim,
                bias=True,
                input_is_parallel=True,
            )
        )

        self.mlp = nn.Sequential(*layers)

    def forward(self, hidden_states):
        """Predict actions from hidden states.

        Args:
            hidden_states: [batch, seq_len, hidden_size]
                          Usually use last token or [ACTION] token hidden state

        Returns:
            [batch, action_dim * horizon] predicted actions
        """
        # Use last token for action prediction
        last_hidden = hidden_states[:, -1, :]  # [batch, hidden_size]

        # Predict actions
        actions = self.mlp(last_hidden)  # [batch, output_dim]

        return actions

    def forward_from_token(
        self,
        hidden_states,
        action_token_positions,
    ):
        """Predict actions from specific token positions.

        Args:
            hidden_states: [batch, seq_len, hidden_size]
            action_token_positions: [batch] indices of [ACTION] tokens

        Returns:
            [batch, action_dim * horizon] predicted actions
        """
        # Gather hidden states at action token positions
        # Expand positions for gathering
        expanded_positions = action_token_positions.unsqueeze(1).expand(-1, hidden_states.size(-1))
        action_hiddens = torch.gather(hidden_states, 1, expanded_positions.unsqueeze(1)).squeeze(1)

        # Predict actions
        actions = self.mlp(action_hiddens)

        return actions

    def reshape_predictions(self, predictions):
        """Reshape flat predictions to [batch, horizon, action_dim].

        Args:
            predictions: [batch, action_dim * horizon]

        Returns:
            [batch, horizon, action_dim]
        """
        batch_size = predictions.size(0)
        return predictions.reshape(batch_size, self.prediction_horizon, self.action_dim)
