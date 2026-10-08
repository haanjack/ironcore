# Copyright (c) 2025-2026 Jaegeun Han
#
# SPDX-License-Identifier: Apache-2.0

"""Standard MLP layer with tensor parallelism and LoRA support.

This module provides the standard MLP layer used in transformer blocks.
It inherits from ParallelMLP and adds LoRA (Low-Rank Adaptation) support.

Async Communication Pattern:
----------------------------
The MLP layer supports async communication for EP (Expert Parallelism):

When async_communication=True, down_proj returns (partial_output, handle) tuple.
Caller must call finalize() to wait for handle and apply bias/dropout.
This allows overlapping expert computation with communication.

trainer.mlp_chunk_size enables checkpointed token-block execution in this layer.
trainer.sequence_chunk_size remains an unimplemented async TP scheduling option.
"""

from typing import Union

import torch
from torch import distributed as dist

from ironcore.config import MainConfig
from ironcore.layers.activations import GLUActivation, get_activation
from ironcore.layers.blockwise import token_chunk_forward
from ironcore.layers.parallel_mlp import ParallelMLP
from ironcore.parallel.random import tensor_parallel_rng_fork
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

        # Store config reference for LoRA
        self.config = config
        self.tensor_model_parallel_size = config.trainer.tensor_model_parallel_size

        # Determine if GLU activation for concatenated weights
        activation = get_activation(model_config.activation_type, model_config.d_model)
        is_glu = isinstance(activation, GLUActivation)
        concatenated_weights = 2 if is_glu else 1

        # Initialize base ParallelMLP with concatenated_weights for GLU
        super().__init__(
            config=config,
            hidden_size=model_config.d_model,
            intermediate_size=model_config.d_ffn,
            gather_output=False,
            name="mlp",
            concatenated_weights=concatenated_weights,
        )

        # Store dropout rate for forward method
        self.dropout_mlp = model_config.dropout_mlp

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
        if self.config.trainer.mlp_chunk_size is not None:
            if async_communication:
                raise ValueError("Block-wise MLP requires synchronous TP communication")
            return token_chunk_forward(
                self._forward_block,
                x,
                self.config.trainer.mlp_chunk_size,
                training=self.training,
            )
        return self._forward_block(x, async_communication)

    def _forward_block(
        self, x: torch.Tensor, async_communication: bool = False
    ) -> Union[torch.Tensor, tuple[torch.Tensor, dist.Work]]:
        """Run dense or LoRA projections on one token block."""
        x = self.up_proj(x)
        x = self.activation(x)
        if async_communication:
            return self.down_proj(x, async_communication=True)

        x = self.down_proj(x)
        if self.dropout_mlp > 0.0:
            with tensor_parallel_rng_fork(self.config.init.seed, x.device):
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

        if self.dropout_mlp > 0.0:
            with tensor_parallel_rng_fork(self.config.init.seed, x.device):
                x = self.dropout(x)
        return x
