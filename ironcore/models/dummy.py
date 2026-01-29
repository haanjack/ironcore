# Copyright (c) 2025-2026 Jaegeun Han
#
# SPDX-License-Identifier: MIT
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the above copyright notice,
# this list of conditions, and the following disclaimer are retained.
#
# Full license text is available at LICENSE file.

from torch import nn
from torch.utils.checkpoint import checkpoint

from ironcore.config import MainConfig
from ironcore.layers.module import BaseModule


class DummyModelLayer(BaseModule):
    def __init__(self, config: MainConfig):
        super().__init__(config)

        self.model_config = config.model
        self.init_config = config.init

        input_size: int = config.model.d_model
        hidden_size: int = config.model.d_ffn
        output_size: int = config.model.d_model

        self.hidden_size = hidden_size

        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def custom_forward(
        self, x, attention_mask, rotary_pos_emb
    ):  # pylint: disable=unused-argument
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def forward(self, hidden_states, attention_mask, rotary_pos_emb):
        return self.custom_forward(hidden_states, attention_mask, rotary_pos_emb)


class DummyModel(BaseModule):

    def __init__(self, config: MainConfig):

        super().__init__(config)

        self.layers = nn.ModuleList(
            [DummyModelLayer(config) for _ in range(config.model.num_layers)]
        )

    def forward(self, hidden_states, attention_mask, rotary_pos_emb):
        for layer in self.layers:
            hidden_states = checkpoint(
                layer.custom_forward,
                hidden_states,
                attention_mask,
                rotary_pos_emb,
                use_reentrant=self.use_reentrant,
            )
        return hidden_states
