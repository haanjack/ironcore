# Copyright (c) 2025-2026 Jaegeun Han
#
# SPDX-License-Identifier: Apache-2.0

"""Base class for parallel MLP layers with tensor parallelism support.

This module provides a common base class for MLP layers that support:
- Tensor parallelism via ColumnParallelLinear and RowParallelLinear
- Async communication for overlapping computation with communication
- GLU activation support

The async communication mechanism allows overlapping EP computation with
communication by returning (partial_output, handle) and requiring finalize()
to be called.

Note: Sequence chunking (sequence_chunk_size) is handled at the transformer
block level, not in this layer.

Usage:
    class MyMLP(ParallelMLP):
        def __init__(self, config, hidden_size, intermediate_size):
            super().__init__(config, hidden_size, intermediate_size)
            # Additional initialization
"""

from typing import Union

import torch
from torch import distributed as dist
from torch import nn

from ironcore.config import MainConfig
from ironcore.layers.activations import GLUActivation, get_activation
from ironcore.layers.module import BaseModule
from ironcore.parallel.tensor_parallel import ColumnParallelLinear, RowParallelLinear


class ParallelMLP(BaseModule):
    """Base class for MLP with tensor parallelism support.

    This class provides the common infrastructure for MLP layers:
    - ColumnParallelLinear for up projection (gather disabled)
    - RowParallelLinear for down projection (input parallel)
    - Activation function with GLU support
    - Optional dropout
    - Async communication support for EP

    Async Communication:
    --------------------
    When async_communication=True in forward():
    - down_proj returns (partial_output, handle) tuple
    - Caller must call finalize() to wait for handle and apply bias/dropout
    - This allows overlapping expert computation with communication

    Note: Sequence chunking is handled at the transformer block level,
    not in this layer.

    Args:
        config: Main configuration
        hidden_size: Input/output hidden dimension
        intermediate_size: FFN intermediate dimension
        gather_output: Whether to gather output (default: False for TP compatibility)
        name: Optional name for debugging (default: None)
        concatenated_weights: Number of concatenated weights in up_proj for GLU (default: 1)
    """

    def __init__(
        self,
        config: MainConfig,
        hidden_size: int,
        intermediate_size: int,
        gather_output: bool = False,
        name: str | None = None,
        concatenated_weights: int = 1,
    ):
        super().__init__(config)

        self.name = name
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size

        model_config = config.model
        self.dropout = nn.Dropout(model_config.dropout_mlp)

        # Get activation function
        self.activation = get_activation(model_config.activation_type, hidden_size)

        # Adjust intermediate size for GLU activations
        d_ffn = intermediate_size
        if isinstance(self.activation, GLUActivation):
            d_ffn = d_ffn * 2

        # Up projection: hidden -> intermediate (ColumnParallel for TP)
        self.up_proj = ColumnParallelLinear(
            config,
            hidden_size,
            d_ffn,
            bias=not model_config.no_bias,
            gather_output=gather_output,
            concatenated_weights=concatenated_weights,
        )

        # Down projection: intermediate -> hidden (RowParallel for TP)
        self.down_proj = RowParallelLinear(
            config,
            intermediate_size,
            hidden_size,
            bias=not model_config.no_bias,
            input_is_parallel=True,  # Input comes from ColumnParallel
        )

        # Store config for dropout check
        self._dropout_rate = model_config.dropout_mlp

    def forward(
        self,
        x: torch.Tensor,
        async_communication: bool = False,
    ) -> Union[torch.Tensor, tuple[torch.Tensor, dist.Work]]:
        """Forward pass through MLP.

        Args:
            x: Input tensor [batch, seq, hidden] or [num_tokens, hidden]
            async_communication: Whether to use async communication for down projection

        Returns:
            If async_communication is False:
                Output tensor with same shape as input
            If async_communication is True:
                Tuple of (partial_output, handle) where finalize() must be called
        """
        # Up projection
        x = self.up_proj(x)

        # Activation
        x = self.activation(x)

        # Down projection (optionally async)
        if async_communication:
            x, handle = self.down_proj(x, async_communication=True)
            return x, handle

        x = self.down_proj(x)

        # Dropout (only in training when rate > 0)
        if self.training and self._dropout_rate > 0.0:
            x = self.dropout(x)

        return x

    def finalize(
        self,
        x: torch.Tensor,
        handle: dist.Work | None,
    ) -> torch.Tensor:
        """Finalize async forward pass.

        Waits for async communication and applies bias/dropout.

        Args:
            x: Partial output from forward()
            handle: Async work handle from forward() (may be None)

        Returns:
            Final output tensor
        """
        if handle is not None:
            handle.wait()

        # Add bias if present (not added in async path)
        if self.down_proj.bias is not None:
            x = x + self.down_proj.bias

        # Dropout (only in training when rate > 0)
        if self.training and self._dropout_rate > 0.0:
            x = self.dropout(x)

        return x

    def __repr__(self):
        name_str = f", name={self.name}" if self.name else ""
        return f"ParallelMLP(hidden={self.hidden_size}, ffn={self.intermediate_size}{name_str})"
