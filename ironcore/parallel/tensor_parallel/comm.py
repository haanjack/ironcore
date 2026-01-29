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
import torch.distributed as dist

from ironcore.parallel import parallel_states


def _reduce(x: torch.Tensor):
    if parallel_states.get_tensor_model_parallel_world_size() == 1:
        return x

    if not x.is_contiguous():
        x = x.contiguous()

    dist.all_reduce(x, group=parallel_states.get_tensor_model_parallel_group())

    return x


def _split_tensor_along_last_dim(x: torch.Tensor):
    world_size = parallel_states.get_tensor_model_parallel_world_size()
    if world_size == 1:
        return x

    # split along last dimension
    assert x.shape[-1] % world_size == 0
    x = x.view(-1, world_size, x.shape[-1] // world_size)

    rank = parallel_states.get_tensor_model_parallel_rank()
    output = x[:, rank].contiguous()

    return output


def _split_concated_tensor_along_last_dim(x: torch.Tensor, num_types: int):
    world_size = parallel_states.get_tensor_model_parallel_world_size()
    if world_size == 1:
        return x

    # split input tensor along the types
    last_dim = x.shape[-1] // num_types
    splited_weights = torch.split(x, last_dim, dim=-1)

    outputs = []
    for splited_weight in splited_weights:
        partition_dim = last_dim // world_size
        partition = torch.split(splited_weight, partition_dim, dim=-1)

        outputs.append(
            partition[parallel_states.get_tensor_model_parallel_rank()])
    output = torch.cat(outputs, dim=-1)

    # # split along last dimension
    # assert x.shape[-1] % world_size == 0
    # x = x.view(-1, world_size, x.shape[-1] // world_size)

    # rank = parallel_states.get_tensor_model_parallel_rank()
    # output = x[:, rank].contiguous()

    return output


def _split_tensor_along_first_dim(x: torch.Tensor):
    world_size = parallel_states.get_tensor_model_parallel_world_size()
    if world_size == 1:
        return x

    # split along first dimension
    assert x.shape[0] % world_size == 0
    x = x.view(world_size, -1, *x.shape[2:])

    rank = parallel_states.get_tensor_model_parallel_rank()
    output = x[rank].contiguous().view(*x.shape[1:])

    return output


def _gather_tensor_along_last_dim(x: torch.Tensor):
    world_size = parallel_states.get_tensor_model_parallel_world_size()
    if world_size == 1:
        return x

    # gather along last dimension
    dim_size = list(x.size())
    dim_size[-1] = x.shape[-1] * world_size

    slices = [torch.empty_like(x, device=x.device) for _ in range(world_size)]
    dist.all_gather(
        slices, x, group=parallel_states.get_tensor_model_parallel_group())

    # Concatenate slices along the last dimension
    output = torch.cat(slices, dim=-1)

    return output


def _gather_concated_tensor_along_last_dim(x: torch.Tensor, num_types: int):
    world_size = parallel_states.get_tensor_model_parallel_world_size()
    if world_size == 1:
        return x

    # gather along last dimension
    dim_size = list(x.size())
    last_dim = x.shape[-1] // num_types
    dim_size[-1] = last_dim * world_size

    # split input tensors along with the last_dim size
    weight_splits = torch.split(x, last_dim, dim=-1)

    outputs = []
    for weight_split in weight_splits:
        slices = [
            torch.empty_like(weight_split, device=weight_split.device)
            for _ in range(world_size)
        ]
        dist.all_gather(
            slices,
            weight_split.contiguous(),
            group=parallel_states.get_tensor_model_parallel_group(),
        )

        # Concatenate slices along the last dimension
        outputs.append(torch.cat(slices, dim=-1))
    output = torch.cat(outputs, dim=-1)

    return output


def _gather_tensor_along_first_dim(x: torch.Tensor):
    world_size = parallel_states.get_tensor_model_parallel_world_size()
    if world_size == 1:
        return x

    # gather along first dimension
    dim_size = list(x.size())
    dim_size[0] *= world_size

    # Gather all slices into the output tensor
    slices = [torch.empty_like(x) for _ in range(world_size)]
    dist.all_gather(
        slices, x, group=parallel_states.get_tensor_model_parallel_group())

    # Concatenate slices along the first dimension
    output = torch.cat(slices, dim=0)

    return output


class _CopyToModelParallelWorkers(
    torch.autograd.Function
):  # pylint: disable=abstract-method
    @staticmethod
    def forward(ctx, x: torch.Tensor):
        return x

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return _reduce(grad_output)


class _ReduceFromModelParallelWorkers(
    torch.autograd.Function
):  # pylint: disable=abstract-method
    @staticmethod
    def forward(ctx, x: torch.Tensor):
        return _reduce(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return grad_output


class _ScatterToModelParallelWorkers(
    torch.autograd.Function
):  # pylint: disable=abstract-method
    @staticmethod
    def forward(ctx, x: torch.Tensor):
        return _split_tensor_along_last_dim(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return _gather_tensor_along_last_dim(grad_output)


class _GatherFromModelParallelWorkers(
    torch.autograd.Function
):  # pylint: disable=abstract-method
    @staticmethod
    def forward(ctx, x: torch.Tensor, attrib: dict):
        ctx.attrib = attrib
        if attrib["column_parallel"]:
            if attrib["concatenated_weights"] > 1:
                return _gather_concated_tensor_along_last_dim(
                    x, attrib["concatenated_weights"]
                )
            else:
                return _gather_tensor_along_last_dim(x)
        elif attrib["row_parallel"]:
            return _gather_tensor_along_first_dim(x)
        return x

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        attrib = ctx.attrib
        if attrib["column_parallel"]:
            if attrib["concatenated_weights"] > 1:
                return _split_concated_tensor_along_last_dim(
                    grad_output, attrib["concatenated_weights"]
                ), None
            else:
                return _split_tensor_along_last_dim(grad_output), None
        elif attrib["row_parallel"]:
            return _split_tensor_along_first_dim(grad_output), None
        return grad_output, None


def copy_inputs_to_model_parallel_workers(x):
    return _CopyToModelParallelWorkers.apply(x)


def reduce_inputs_from_model_parallel_workers(x):
    return _ReduceFromModelParallelWorkers.apply(x)


def scatter_input_to_model_parallel_workers(x):
    return _ScatterToModelParallelWorkers.apply(x)


def gather_from_model_parallel_workers(x, attrib):
    return _GatherFromModelParallelWorkers.apply(x, attrib)


def split_to_model_parallel_workers(x, attrib):
    if attrib["column_parallel"]:
        if attrib["concatenated_weights"] > 1:
            return _split_concated_tensor_along_last_dim(
                x, attrib["concatenated_weights"]
            )
        else:
            return _split_tensor_along_last_dim(x)
    elif attrib["row_parallel"]:
        return _split_tensor_along_first_dim(x)
