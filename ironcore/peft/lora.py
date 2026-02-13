# Copyright (c) 2025-2026 Jaegeun Han
#
# SPDX-License-Identifier: MIT
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the above copyright notice,
# this list of conditions, and the following disclaimer are retained.
#
# Full license text is available at LICENSE file.

import math

import torch
import torch.nn as nn

from ironcore.config import LoRAConfig


class LoRALinear(nn.Module):
    """
    Base LoRA adapter.

    Implements low-rank adaptation: h = (B @ A)(x) * scaling
    where A is initialized with Kaiming uniform and B is initialized with zeros.

    Args:
        in_features: Input dimension
        out_features: Output dimension
        rank: LoRA rank (r)
        alpha: LoRA scaling parameter
        dropout: Dropout probability for LoRA activations
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int,
        alpha: float,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        # LoRA matrices: A (in -> rank), B (rank -> out)
        self.lora_A = nn.Parameter(torch.zeros(in_features, rank))
        self.lora_B = nn.Parameter(torch.zeros(rank, out_features))

        # Dropout (optional)
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else None

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize LoRA weights following standard practice."""
        # A: Kaiming uniform (ensures gradient flow)
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        # B: zeros (ensures LoRA starts as identity - no effect initially)
        nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through LoRA adapter.

        Args:
            x: Input tensor [batch, seq, in_features]

        Returns:
            LoRA output tensor [batch, seq, out_features]
        """
        # x @ A: [batch, seq, in_features] @ [in_features, rank] -> [batch, seq, rank]
        result = torch.matmul(x, self.lora_A)

        # Apply dropout to intermediate activations if configured
        if self.dropout is not None:
            result = self.dropout(result)

        # result @ B: [batch, seq, rank] @ [rank, out_features] -> [batch, seq, out_features]
        result = torch.matmul(result, self.lora_B)

        # Apply scaling
        return self.scaling * result

    def __repr__(self):
        return (
            f"LoRALinear(in_features={self.in_features}, "
            f"out_features={self.out_features}, rank={self.rank}, "
            f"alpha={self.alpha}, scaling={self.scaling:.4f})"
        )


class LoRAColumnParallelLinear(nn.Module):
    """
    LoRA wrapper for ColumnParallelLinear with sharded adapters.

    In column-parallel layers, the output dimension is sharded across TP ranks.
    We shard the LoRA adapter's B matrix to match the base layer's partition.
    This makes LoRA computation truly parallel and more memory efficient.

    Args:
        base_layer: The underlying ColumnParallelLinear layer
        lora_config: LoRA configuration
    """

    def __init__(self, base_layer, lora_config: LoRAConfig):
        super().__init__()
        self.base_layer = base_layer

        # LoRA A is replicated (shared across all TP ranks)
        # LoRA B is sharded (each TP rank has a slice of the output dim)
        self.tp_rank = base_layer.tensor_model_parallel_rank
        self.tp_size = base_layer.tensor_model_parallel_size
        self.output_size_per_partition = base_layer.output_size

        self.lora = LoRALinear(
            in_features=base_layer.input_size,
            out_features=self.output_size_per_partition,
            rank=lora_config.r,
            alpha=lora_config.alpha,
            dropout=lora_config.dropout,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with sharded LoRA.

        Args:
            x: Input tensor [batch, seq, in_features]

        Returns:
            Combined output [batch, seq, out_features_per_partition]
        """
        # Base computation (already sharded)
        base_output = self.base_layer(x)  # [batch, seq, out_features/tp_size]

        # LoRA computation (already sharded by self.lora being local size)
        lora_output = self.lora(x)  # [batch, seq, out_features/tp_size]

        # Combine base and LoRA (both are sharded same way)
        return base_output + lora_output

    def __repr__(self):
        return f"LoRAColumnParallelLinear(\n  {self.base_layer}\n  {self.lora}\n)"


class LoRAConcatenatedColumnParallel(nn.Module):
    """
    LoRA wrapper for ColumnParallelLinear with concatenated_weights > 1.

    Each concatenated portion (e.g., K, V) gets its own LoRA adapter,
    which is sharded matching the base layer's TP partition.
    """

    def __init__(
        self,
        base_layer,
        lora_config: LoRAConfig,
        target_modules: list[str],
    ):
        super().__init__()
        self.base_layer = base_layer

        if base_layer.concatenated_weights <= 1:
            raise ValueError(
                "LoRAConcatenatedColumnParallel requires concatenated_weights > 1"
            )

        self.num_concatenated = base_layer.concatenated_weights
        self.output_size_per_concat = base_layer.output_size // self.num_concatenated
        
        self.tp_rank = base_layer.tensor_model_parallel_rank
        self.tp_size = base_layer.tensor_model_parallel_size

        # Create separate LoRA for each target module
        self.lora_adapters = nn.ModuleList()
        self.adapter_map = {}  # index -> adapter_idx in lora_adapters

        for i in range(self.num_concatenated):
            # Check if this index should have LoRA
            name = target_modules[i] if i < len(target_modules) else None
            if name and name in lora_config.target_modules:
                # LoRA adapter is sharded to local size
                adapter = LoRALinear(
                    in_features=base_layer.input_size,
                    out_features=self.output_size_per_concat,
                    rank=lora_config.r,
                    alpha=lora_config.alpha,
                    dropout=lora_config.dropout,
                )
                self.lora_adapters.append(adapter)
                self.adapter_map[i] = len(self.lora_adapters) - 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with sharded LoRA for concatenated weights.
        """
        # Base computation
        base_output = self.base_layer(x)  # [batch, seq, total_out/tp_size]

        # Split base output by concatenated portions
        base_splits = torch.split(base_output, self.output_size_per_concat, dim=-1)

        # Apply LoRA to targeted portions
        combined_splits = []
        for i in range(self.num_concatenated):
            base_shard = base_splits[i]

            if i in self.adapter_map:
                # This portion has a local LoRA adapter shard
                adapter = self.lora_adapters[self.adapter_map[i]]
                lora_shard = adapter(x)
                combined_splits.append(base_shard + lora_shard)
            else:
                combined_splits.append(base_shard)

        # Concatenate back
        return torch.cat(combined_splits, dim=-1)

    def __repr__(self):
        return (
            f"LoRAConcatenatedColumnParallel(concatenated_weights={self.num_concatenated},\n"
            f"  {self.base_layer}\n"
            f"  {len(self.lora_adapters)} LoRA adapters\n)"
        )


class LoRARowParallelLinear(nn.Module):
    """
    LoRA wrapper for RowParallelLinear with sharded adapters.

    In row-parallel layers, the input dimension is sharded.
    We shard the LoRA adapter's A matrix to match the input partition.
    The LoRA contribution is added after the base layer's all-reduce
    (via finalize in async mode or directly in sync mode).
    """

    def __init__(self, base_layer, lora_config: LoRAConfig):
        super().__init__()
        self.base_layer = base_layer

        # LoRA A is sharded (matching input partition)
        # LoRA B is replicated
        self.lora = LoRALinear(
            in_features=base_layer.input_size, # Already sharded size
            out_features=base_layer.output_size,
            rank=lora_config.r,
            alpha=lora_config.alpha,
            dropout=lora_config.dropout,
        )

    def forward(self, x: torch.Tensor, async_communication: bool = False):
        """
        Forward pass with sharded LoRA.
        """
        # 1. Base forward (returns partial result if async)
        if async_communication:
            base_partial, handle = self.base_layer(x, async_communication=True)
            lora_partial = self.lora(x)
            return (base_partial, lora_partial), handle

        # Sync path: combine base and lora before all-reduce
        from ironcore.parallel.tensor_parallel import comm
        if self.base_layer.input_is_parallel:
            parallel_x = x
        else:
            parallel_x = comm.scatter_input_to_model_parallel_workers(x)
        
        base_partial = torch.matmul(parallel_x, self.base_layer.weight)
        lora_partial = self.lora(x)
        
        combined_partial = base_partial + lora_partial
        output = comm.reduce_inputs_from_model_parallel_workers(combined_partial)
        
        if self.base_layer.bias is not None:
            output = output + self.base_layer.bias
        return output

    def finalize(self, outputs: tuple, handle):
        """
        Complete async operation by waiting for all-reduce and adding components.
        """
        base_partial, lora_partial = outputs

        # Wait for base layer's all-reduce to complete (this was for base_partial only)
        if handle is not None:
            handle.wait()

        # Add LoRA partial contribution (needs its own reduce if async was separate)
        from ironcore.parallel.tensor_parallel import comm
        lora_output_reduced = comm.reduce_inputs_from_model_parallel_workers(lora_partial)

        # Add bias to base (base layer doesn't add it in async mode)
        if self.base_layer.bias is not None:
            base_partial = base_partial + self.base_layer.bias

        # Add LoRA contribution
        return base_partial + lora_output_reduced

    def __repr__(self):
        return f"LoRARowParallelLinear(\n  {self.base_layer}\n  {self.lora}\n)"
