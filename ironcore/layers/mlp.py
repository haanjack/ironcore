# Copyright (c) 2025-2026 Jaegeun Han
#
# SPDX-License-Identifier: MIT
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the above copyright notice,
# this list of conditions, and the following disclaimer are retained.
#
# Full license text is available at LICENSE file.

"""Standard MLP layer with tensor parallelism and LoRA support.

This module provides the standard MLP layer used in transformer blocks.
It inherits from ParallelMLP and adds LoRA (Low-Rank Adaptation) support.

Async Communication Pattern:
----------------------------
The MLP layer supports async communication for EP (Expert Parallelism):

When async_communication=True, down_proj returns (partial_output, handle) tuple.
Caller must call finalize() to wait for handle and apply bias/dropout.
This allows overlapping expert computation with communication.

Note: Sequence chunking (sequence_chunk_size) is handled at the transformer
block level, not in this layer. The transformer splits sequences and calls
MLP on each chunk separately.
"""

from typing import Union

import torch
from torch import distributed as dist

from ironcore.config import MainConfig
from ironcore.layers.activations import GLUActivation, get_activation
from ironcore.layers.parallel_mlp import ParallelMLP
from ironcore.parallel.tensor_parallel import ColumnParallelLinear, RowParallelLinear
from ironcore.peft import wrap_with_lora_if_target


class MLP(ParallelMLP):
    """Standard MLP layer for transformer blocks.

    Inherits from ParallelMLP and adds:
    - LoRA (Low-Rank Adaptation) support for PEFT
    - Uses model d_model and d_ffn from config

    This is the main MLP class used in transformer layers.

    Args:
        config: Main configuration containing model settings
    """

    def __init__(self, config: MainConfig):
        model_config = config.model

        # Initialize base ParallelMLP
        super().__init__(
            config=config,
            hidden_size=model_config.d_model,
            intermediate_size=model_config.d_ffn,
            gather_output=False,
            name="mlp",
        )

        # Store additional references for LoRA
        self.config = model_config
        self.tensor_model_parallel_size = config.trainer.tensor_model_parallel_size

        # Re-get activation to determine if GLU (for LoRA wrapping)
        self.activation = get_activation(model_config.activation_type, model_config.d_model)

        # Recalculate d_ffn for GLU
        is_glu = isinstance(self.activation, GLUActivation)
        d_ffn = model_config.d_ffn
        if is_glu:
            d_ffn = d_ffn * 2

        # Re-create up_proj and down_proj with correct settings
        # (needed because we need to wrap with LoRA after creation)
        self.up_proj = ColumnParallelLinear(
            config,
            model_config.d_model,
            d_ffn,
            bias=not model_config.no_bias,
            concatenated_weights=2 if is_glu else 1,
        )
        self.down_proj = RowParallelLinear(
            config,
            model_config.d_ffn,
            model_config.d_model,
            bias=not model_config.no_bias,
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

    def forward(
        self,
        x: torch.Tensor,
        async_communication: bool = False,
    ) -> Union[torch.Tensor, tuple[torch.Tensor, dist.Work]]:
        """Forward pass through MLP.

        Args:
            x: Input tensor
            async_communication: Whether to use async communication for down projection

        Returns:
            If async_communication is False:
                Output tensor
            If async_communication is True:
                Tuple of (partial_output, handle) where finalize() must be called
        """
        x = self.up_proj(x)
        x = self.activation(x)
        if async_communication:
            return self.down_proj(x, async_communication=True)

        x = self.down_proj(x)
        if self.config.dropout_mlp > 0.0:
            x = self.dropout(x)
        return x

    def finalize(
        self,
        x: torch.Tensor,
        handle: dist.Work | None,
    ) -> torch.Tensor:
        """Finalize async forward pass.

        Waits for async communication and applies bias/dropout.
        Handles both LoRA-wrapped and standard down_proj.

        Args:
            x: Partial output from forward()
            handle: Async work handle from forward()

        Returns:
            Final output tensor
        """
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
