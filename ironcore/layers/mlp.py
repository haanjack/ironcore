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
from ironcore.peft import wrap_with_lora_if_target


class MLP(BaseModule):
    def __init__(self, config: MainConfig):
        super().__init__(config)

        self.config = config.model
        self.tensor_model_parallel_size = config.trainer.tensor_model_parallel_size

        self.activation = get_activation(config.model.activation_type, self.config.d_model)
        self.dropout = nn.Dropout(config.model.dropout_mlp)

        d_ffn = config.model.d_ffn
        is_glu = isinstance(self.activation, GLUActivation)
        if is_glu:
            d_ffn = d_ffn * 2
        self.up_proj = ColumnParallelLinear(
            config,
            self.config.d_model,
            d_ffn,
            bias=not config.model.no_bias,
            concatenated_weights=2 if is_glu else 1,
        )
        self.down_proj = RowParallelLinear(
            config,
            config.model.d_ffn,
            self.config.d_model,
            bias=not config.model.no_bias,
            input_is_parallel=True,
        )

        # Wrap with LoRA if PEFT is enabled
        if config.peft.method == "lora":
            if is_glu:
                # Up and Gate are concatenated in up_proj
                self.up_proj = wrap_with_lora_if_target(
                    self.up_proj, ["up_proj", "gate_proj"], config.peft.lora, concatenated=True
                )
            else:
                self.up_proj = wrap_with_lora_if_target(self.up_proj, "up_proj", config.peft.lora)

            self.down_proj = wrap_with_lora_if_target(self.down_proj, "down_proj", config.peft.lora)

    def forward(self, x, async_communication=False):
        x = self.up_proj(x)
        x = self.activation(x)
        if async_communication:
            return self.down_proj(x, async_communication=True)

        x = self.down_proj(x)
        if self.config.dropout_mlp > 0.0:
            x = self.dropout(x)
        return x

    def finalize(self, x, handle):
        # Handle LoRA-wrapped down_proj
        if hasattr(self.down_proj, "finalize"):
            # LoRA-wrapped layer handles finalization internally
            x = self.down_proj.finalize(x, handle)
        else:
            # Standard path without LoRA
            if handle:
                handle.wait()

            if self.down_proj.bias is not None:
                x = x + self.down_proj.bias

        if self.config.dropout_mlp > 0.0:
            x = self.dropout(x)
        return x
