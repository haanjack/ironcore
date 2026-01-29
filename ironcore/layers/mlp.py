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

from ironcore.config import MainConfig
from ironcore.layers.activations import GLUActivation, get_activation
from ironcore.layers.module import BaseModule
from ironcore.parallel.tensor_parallel import ColumnParallelLinear, RowParallelLinear


class MLP(BaseModule):
    def __init__(self, config: MainConfig):
        super().__init__(config)

        self.config = config.model
        self.tensor_model_parallel_size = config.trainer.tensor_model_parallel_size

        self.activation = get_activation(
            config.model.activation_type, self.config.d_model
        )
        self.dropout = nn.Dropout(config.model.dropout_mlp)

        d_ffn = config.model.d_ffn
        if isinstance(self.activation, GLUActivation):
            d_ffn = d_ffn * 2
        self.up_proj = ColumnParallelLinear(
            config,
            self.config.d_model,
            d_ffn,
            bias=not config.model.no_bias,
        )
        self.down_proj = RowParallelLinear(
            config,
            config.model.d_ffn,
            self.config.d_model,
            bias=not config.model.no_bias,
            input_is_parallel=True,
        )

    def forward(self, x):
        x = self.up_proj(x)
        x = self.activation(x)
        x = self.down_proj(x)
        if self.config.dropout_mlp > 0.0:
            x = self.dropout(x)
        return x
