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
from torch.functional import F

from ironcore.layers.module import BaseModule
from ironcore.parallel import parallel_states
from ironcore.parallel.tensor_parallel import comm


class ParallelLinear(BaseModule):  # pylint: disable=abstract-method
    def __init__(
        self,
        config,
        input_size: int,
        output_size: int,
        bias: bool = False,
    ):
        super().__init__(config)
        self.input_size = input_size
        self.output_size = output_size
        self.use_bias = bias
        self.tensor_model_parallel_size = config.trainer.tensor_model_parallel_size

        self.tensor_model_parallel_rank = 1
        if self.tensor_model_parallel_size > 1:
            self.tensor_model_parallel_rank = (
                parallel_states.get_tensor_model_parallel_rank()
            )

        # Divide the weight matrix
        self.weight = nn.Parameter(torch.Tensor(input_size, output_size))
        if self.use_bias:
            self.bias = nn.Parameter(torch.Tensor(output_size))
        else:
            self.bias = None

        self.column_parallel = False
        self.row_parallel = False
        self.concatenated_weights = 1


class VocabParallelEmbedding(ParallelLinear):
    """
    Embedding layer with tensor model parallelism.
    The embedding layer is defined as Y = XB + b. B is parallelized along the vocab size.

    Args:
        config (Config): configuration for the model
        num_tokens (int): size of the vocabulary
        hidden_size (int): size of the hidden state
    """

    def __init__(
        self,
        config,
        input_dim: int,
        embedding_dim: int,
        padding_start_idx: int = None,
        parallel_input: bool = False,
        parallel_output: bool = False,
    ):

        if input_dim % parallel_states.get_tensor_model_parallel_world_size() != 0:
            raise ValueError(
                "The vocabulary size ({}) must be evenly divisible by the tensor model parallel size"
            )
        self.parallel_input_dim = (
            input_dim // parallel_states.get_tensor_model_parallel_world_size()
        )
        super().__init__(
            config, input_size=self.parallel_input_dim, output_size=embedding_dim
        )

        self.padding_start_idx = padding_start_idx
        self.parallel_input = parallel_input
        self.parallel_output = parallel_output
        self.row_parallel = True

        # we will calculate local_padding_start_idx later in init_weight()
        self.local_padding_start_idx = None

    def init_weight(self):
        """initialize embedding weights"""
        super().init_weights()

        # set local padding start idx
        if self.padding_start_idx is not None:
            # calcualte local padding index for this rank
            start_idx = (
                self.parallel_input_dim
                * parallel_states.get_tensor_model_parallel_rank()
            )
            end_idx = self.parallel_input_dim * (
                parallel_states.get_tensor_model_parallel_rank() + 1
            )

            local_padding = (self.padding_start_idx >= start_idx) & (
                self.padding_start_idx < end_idx
            )
            self.local_padding_start_idx = (
                self.padding_start_idx - start_idx
                if local_padding
                else end_idx - start_idx
            )

            # zero out from local_padding_start_idx to the end
            with torch.no_grad():
                self.weight[self.local_padding_start_idx:].zero_()

        def _zero_padding_grad_hook(grad):
            # zero out the gradient in the padding region to disable updates during training
            grad[self.local_padding_start_idx:] = 0
            return grad

        # register hook
        self.weight.register_hook(_zero_padding_grad_hook)

    def forward(self, x):
        """
        Forward pass for the embedding layer.
        """

        start_idx = (
            self.parallel_input_dim * parallel_states.get_tensor_model_parallel_rank()
        )
        end_idx = self.parallel_input_dim * (
            parallel_states.get_tensor_model_parallel_rank() + 1
        )

        # set token ids to the corresponding embedding space
        token_mask = (x >= start_idx) & (x < end_idx)
        x_partition = (x - start_idx) * token_mask

        x_partition = F.embedding(
            x_partition.to(device=self.weight.device), self.weight
        )
        
        # Mask out embeddings for tokens not in this partition
        x_partition[~token_mask, :] = 0.0

        if parallel_states.get_tensor_model_parallel_world_size() > 1:
            x = comm.reduce_inputs_from_model_parallel_workers(x_partition)
        else:
            x = x_partition
        return x


class ColumnParallelLinear(ParallelLinear):
    """
    Linear layer with column parallelism.
    The linear layer is defined as Y = XA + b. A is parallelized along
    its second dimension as A = [A_1, ..., A_p].

    Args:
        input_size: first dimension of matrix A
        output_size: second dimension of matrix A
        bias: If true, add bias
        input_is_parallel: If true, we assume that the input is already
                           split across the GPUs and we do not split
                           again.
        init_method: method to initialize weights
    """

    def __init__(
        self,
        config,
        input_size: int,
        output_size: int,
        bias: bool = True,
        gather_output=False,
        concatenated_weights: int = 1,
    ):

        self.tensor_model_parallel_size = config.trainer.tensor_model_parallel_size
        assert (
            output_size % self.tensor_model_parallel_size == 0
        ), "output_size must be divisible by tensor_model_parallel_size in ColumnParallelLinear"
        output_size = output_size // self.tensor_model_parallel_size
        self.gather_output = gather_output
        super().__init__(config, input_size, output_size, bias)
        self.column_parallel = True
        self.concatenated_weights = concatenated_weights

    def forward(self, x):
        parallel_x = comm.copy_inputs_to_model_parallel_workers(x)
        parallel_output = torch.matmul(parallel_x, self.weight)
        if self.use_bias:
            parallel_output = parallel_output + self.bias
        if self.gather_output:
            output = comm.gather_from_model_parallel_workers(
                parallel_output, {"column_parallel": True, "concatenated_weights": self.concatenated_weights})
        else:
            output = parallel_output
        return output

    def __repr__(self):
        return f"{self.__class__.__name__}({self.input_size}, {self.output_size})"


class RowParallelLinear(ParallelLinear):
    """
    Linear layer with row parallelism.
    The linear layer is defined as Y = XA + b. A is parallelized along
    its first dimension and X along its second dimension as:
               -   -
              | A_1 |
              | .   |
          A = | .   |        X = [X_1, ..., X_p]
              | .   |
              | A_p |
               -   -

    Args:
        input_size: first dimension of matrix A
        output_size: second dimension of matrix A
        bias: If true, add bias
        input_is_parallel: If true, we assume that the input is already
                           split across the GPUs and we do not split
                           again.
        init_method: method to initialize weights
    """

    def __init__(
        self,
        config,
        input_size: int,
        output_size: int,
        bias: bool = True,
        input_is_parallel: bool = False,
    ):
        self.input_is_parallel = input_is_parallel
        self.tensor_model_parallel_size = config.trainer.tensor_model_parallel_size
        assert (
            input_size % self.tensor_model_parallel_size == 0
        ), "input_size must be divisible by tensor_model_parallel_size in RowParallelLinear"
        input_size = input_size // self.tensor_model_parallel_size
        super().__init__(config, input_size, output_size, bias)
        self.row_parallel = True

    def forward(self, x):
        if self.input_is_parallel:
            parallel_x = x
        else:
            parallel_x = comm.scatter_input_to_model_parallel_workers(x)
        output = torch.matmul(parallel_x, self.weight)

        if self.tensor_model_parallel_size > 1:
            output = comm.reduce_inputs_from_model_parallel_workers(output)

        if self.use_bias:
            output = output + self.bias

        return output
