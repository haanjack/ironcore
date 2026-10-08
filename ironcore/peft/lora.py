# Copyright (c) 2025-2026 Jaegeun Han
#
# SPDX-License-Identifier: Apache-2.0

import math

import torch
from torch import nn

from ironcore.config import LoRAConfig
from ironcore.parallel.random import tensor_parallel_rng_fork


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
        # Adapter parameters are full replicas. Only transient computation views
        # are partitioned at the column/row-parallel boundaries.
        self.column_parallel = False
        self.row_parallel = False
        self.concatenated_weights = 1
        for parameter in (self.lora_A, self.lora_B):
            parameter.is_tp_sharded = False
            parameter.tp_shard_dim = None

        # Dropout (optional)
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else None

        # Initialize weights
        self._init_weights()

    def _init_weights(self, generator: torch.Generator | None = None) -> None:
        """Initialize LoRA weights following standard practice."""
        # A: Kaiming uniform (ensures gradient flow)
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5), generator=generator)
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

    def _dropout(self, hidden: torch.Tensor, seed: int) -> torch.Tensor:
        if self.dropout is not None and self.training:
            with tensor_parallel_rng_fork(seed, hidden.device):
                hidden = self.dropout(hidden)
        return hidden

    def forward_column(self, x: torch.Tensor, seed: int) -> torch.Tensor:
        """Split B's compute view; sum low-rank gradients before replicated A."""
        from ironcore.parallel.tensor_parallel import comm

        hidden = self._dropout(x @ self.lora_A, seed)
        hidden = comm.copy_inputs_to_model_parallel_workers(hidden)
        local_b = comm.scatter_input_to_model_parallel_workers(self.lora_B)
        return self.scaling * (hidden @ local_b)

    def forward_row(self, x: torch.Tensor, seed: int) -> torch.Tensor:
        """Reduce partial low-rank activations before dropout and replicated B."""
        from ironcore.parallel.tensor_parallel import comm

        local_a = comm.scatter_input_to_model_parallel_workers(self.lora_A.T).T
        hidden = comm.reduce_inputs_from_model_parallel_workers(x @ local_a)
        hidden = self._dropout(hidden, seed)
        return self.scaling * (hidden @ self.lora_B)

    def __repr__(self):
        return (
            f"LoRALinear(in_features={self.in_features}, "
            f"out_features={self.out_features}, rank={self.rank}, "
            f"alpha={self.alpha}, scaling={self.scaling:.4f})"
        )


class LoRAColumnParallelLinear(nn.Module):
    """
    LoRA wrapper for ColumnParallelLinear with replicated adapters.

    In column-parallel layers, the output dimension is sharded across TP ranks.
    A and B remain full parameters. B's temporary compute view follows the
    base projection's output partition; backward gathers B and sums A gradients.

    Args:
        base_layer: The underlying ColumnParallelLinear layer
        lora_config: LoRA configuration
    """

    def __init__(self, base_layer, lora_config: LoRAConfig):
        super().__init__()
        self.base_layer = base_layer

        # Both adapter matrices are replicated across TP ranks.
        self.tp_rank = base_layer.tensor_model_parallel_rank
        self.tp_size = base_layer.tensor_model_parallel_size
        self.output_size_per_partition = base_layer.output_size

        self.lora = LoRALinear(
            in_features=base_layer.input_size,
            out_features=self.output_size_per_partition * self.tp_size,
            rank=lora_config.r,
            alpha=lora_config.alpha,
            dropout=lora_config.dropout,
        )
        # Expose attributes for checkpointing logic
        self.column_parallel = True
        self.row_parallel = False
        self.concatenated_weights = base_layer.concatenated_weights

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with replicated adapters and a local output partition.

        Args:
            x: Input tensor [batch, seq, in_features]

        Returns:
            Combined output [batch, seq, out_features_per_partition]
        """
        # Base computation (already sharded)
        base_output = self.base_layer(x)  # [batch, seq, out_features/tp_size]

        lora_output = self.lora.forward_column(x, self.base_layer.config.init.seed)
        if self.base_layer.gather_output:
            from ironcore.parallel.tensor_parallel import comm

            lora_output = comm.gather_from_model_parallel_workers(
                lora_output, {"column_parallel": True}
            )

        # Combine base and LoRA (both are sharded same way)
        return base_output + lora_output

    def __repr__(self):
        return f"LoRAColumnParallelLinear(\n  {self.base_layer}\n  {self.lora}\n)"


class LoRAConcatenatedColumnParallel(nn.Module):
    """
    Replicated adapters for ColumnParallelLinear with concatenated_weights > 1.

    Each concatenated portion (e.g., K, V) gets its own full adapter; only
    its output computation is split to match the base layer's TP partition.
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
            raise ValueError("LoRAConcatenatedColumnParallel requires concatenated_weights > 1")

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
                # Keep the full adapter and partition its computation at runtime.
                adapter = LoRALinear(
                    in_features=base_layer.input_size,
                    out_features=self.output_size_per_concat * self.tp_size,
                    rank=lora_config.r,
                    alpha=lora_config.alpha,
                    dropout=lora_config.dropout,
                )
                self.lora_adapters.append(adapter)
                self.adapter_map[i] = len(self.lora_adapters) - 1

        # Expose attributes for checkpointing logic
        self.column_parallel = True
        self.row_parallel = False
        self.concatenated_weights = base_layer.concatenated_weights

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with replicated adapters for concatenated projections.
        """
        # Base computation
        base_output = self.base_layer(x)  # [batch, seq, total_out/tp_size]

        # Apply LoRA to targeted portions
        combined_splits = []
        for i in range(self.num_concatenated):
            if i in self.adapter_map:
                adapter = self.lora_adapters[self.adapter_map[i]]
                combined_splits.append(adapter.forward_column(x, self.base_layer.config.init.seed))
            else:
                combined_splits.append(x.new_zeros(*x.shape[:-1], self.output_size_per_concat))

        # Concatenate back
        delta = torch.cat(combined_splits, dim=-1)
        if self.base_layer.gather_output:
            from ironcore.parallel.tensor_parallel import comm

            delta = comm.gather_from_model_parallel_workers(
                delta, {"column_parallel": True, "concatenated_weights": self.num_concatenated}
            )
        return base_output + delta

    def __repr__(self):
        return (
            f"LoRAConcatenatedColumnParallel(concatenated_weights={self.num_concatenated},\n"
            f"  {self.base_layer}\n"
            f"  {len(self.lora_adapters)} LoRA adapters\n)"
        )


class LoRARowParallelLinear(nn.Module):
    """
    LoRA wrapper for RowParallelLinear with replicated adapters.

    In row-parallel layers, the input dimension is sharded.
    A's temporary compute view follows the input partition. Its low-rank
    activation is reduced before multiplying B, preserving full adapter gradients.
    """

    def __init__(self, base_layer, lora_config: LoRAConfig):
        super().__init__()
        self.base_layer = base_layer

        # Both matrices stay replicated; forward_row partitions A's compute view.
        self.lora = LoRALinear(
            in_features=base_layer.input_size * base_layer.tensor_model_parallel_size,
            out_features=base_layer.output_size,
            rank=lora_config.r,
            alpha=lora_config.alpha,
            dropout=lora_config.dropout,
        )

        # Expose attributes for checkpointing logic
        self.column_parallel = False
        self.row_parallel = True
        self.concatenated_weights = 1

    def forward(self, x: torch.Tensor, async_communication: bool = False):
        """
        Forward pass with replicated adapters and input partitions.
        """
        from ironcore.parallel.tensor_parallel import comm

        if self.base_layer.input_is_parallel:
            parallel_x = x
        else:
            parallel_x = comm.scatter_input_to_model_parallel_workers(x)

        base_partial = torch.matmul(parallel_x, self.base_layer.weight)
        output = comm.reduce_inputs_from_model_parallel_workers(base_partial)
        output = output + self.lora.forward_row(parallel_x, self.base_layer.config.init.seed)

        if async_communication:
            # Collectives stay autograd-aware. Bias is added by finalize(),
            # matching the existing synchronous fallback's async interface.
            return output, None

        if self.base_layer.bias is not None:
            output = output + self.base_layer.bias
        return output

    def finalize(self, output: torch.Tensor, handle):
        """
        Complete async operation by waiting for all-reduce and adding bias.
        """
        # Wait for the single combined all-reduce to complete
        if handle is not None:
            handle.wait()

        # Add bias to the final result
        if self.base_layer.bias is not None:
            output = output + self.base_layer.bias

        return output

    def __repr__(self):
        return f"LoRARowParallelLinear(\n  {self.base_layer}\n  {self.lora}\n)"
