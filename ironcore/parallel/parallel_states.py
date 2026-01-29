# Copyright (c) 2025-2026 Jaegeun Han
#
# SPDX-License-Identifier: MIT
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the above copyright notice,
# this list of conditions, and the following disclaimer are retained.
#
# Full license text is available at LICENSE file.

import os
from datetime import timedelta

import torch
from torch import distributed as dist

# parallel world size
_DATA_PARALLEL_WORLD_SIZE = None
_TENSOR_MODEL_PARALLEL_WORLD_SIZE = None

# parallel groups
_DATA_PARALLEL_GROUP = None

_TENSOR_MODEL_PARALLEL_GROUP = None


def initialize_model_parallel(
    tensor_model_parallel_size: int,
    timeout_in_minutes: float,
):
    """Initialize parallel groups for model parallel communication"""
    # pylint: disable=global-statement

    global _TENSOR_MODEL_PARALLEL_WORLD_SIZE
    _TENSOR_MODEL_PARALLEL_WORLD_SIZE = tensor_model_parallel_size

    global _DATA_PARALLEL_WORLD_SIZE
    _DATA_PARALLEL_WORLD_SIZE = (
        int(os.getenv("WORLD_SIZE", "1")) // tensor_model_parallel_size
    )

    if not dist.is_initialized():
        return

    rank = dist.get_rank()
    world_size = dist.get_world_size()

    assert (
        world_size % tensor_model_parallel_size == 0
    ), "world size must be divisible by tensor model parallel size"

    timeout = timedelta(minutes=timeout_in_minutes)
    backend = "nccl" if torch.cuda.is_available() else "gloo"

    dp_world_size = world_size // tensor_model_parallel_size

    # Initialize the ranks for the tensor model parallel groups.
    #
    # The tensor model parallel group is used to perform all-gather/scatter operations over the tensor model parallel dimension.
    # The operation is performed within the same data parallel group.
    #
    # For example, if dp_world_size is 4 and tensor_model_parallel_size is 4,
    # tp_ranks would be [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11], [12, 13, 14, 15]].
    global _TENSOR_MODEL_PARALLEL_GROUP
    tp_ranks = [
        [i * tensor_model_parallel_size +
            j for j in range(tensor_model_parallel_size)]
        for i in range(dp_world_size)
    ]
    for tp_group_id, ranks in enumerate(tp_ranks):
        group = dist.new_group(
            ranks,
            timeout=timeout,
            backend=backend,
        )
        # group_desc=f"tensor parallel group ({tp_group_id})",
        if rank in ranks:
            _TENSOR_MODEL_PARALLEL_GROUP = group

    # Initialize the ranks for the data parallel groups.
    #
    # The data parallel group is used to perform all-reduce operations over the data parallel dimension.
    # The operation is performed within the same model parallel group.
    #
    # For example, if tensor_model_parallel_size is 4 and world_size is 16,
    # dp_ranks would be [[0, 4, 8, 12], [1, 5, 9, 13], [2, 6, 10, 14], [3, 7, 11, 15]].
    global _DATA_PARALLEL_GROUP
    dp_ranks = [
        [tp_group[i] for tp_group in tp_ranks]
        for i in range(tensor_model_parallel_size)
    ]
    # create a new process group for data parallelism
    for dp_group_id, ranks in enumerate(dp_ranks):
        group = dist.new_group(
            ranks,
            timeout=timeout,
            backend=backend,
        )
        # group_desc=f"data parallel group ({dp_group_id})",
        if rank in ranks:
            _DATA_PARALLEL_GROUP = group


def get_data_parallel_world_size():
    """Get data parallel world size."""
    assert (
        _DATA_PARALLEL_WORLD_SIZE is not None
    ), "data parallel world size should not be None"
    return _DATA_PARALLEL_WORLD_SIZE


def get_tensor_model_parallel_world_size():
    """Get tensor model parallel world size."""
    assert (
        _TENSOR_MODEL_PARALLEL_WORLD_SIZE is not None
    ), "tensor model parallel world size should not be None"
    return _TENSOR_MODEL_PARALLEL_WORLD_SIZE


def get_tensor_model_parallel_group():
    """Get model parallel group that the caller rank belongs to."""
    assert (
        _TENSOR_MODEL_PARALLEL_GROUP is not None
    ), "model parallel group should not be None"
    return _TENSOR_MODEL_PARALLEL_GROUP


def get_data_parallel_group():
    """Get data parallel group that the caller rank belongs to."""
    assert _DATA_PARALLEL_GROUP is not None, "data parallel group should not be None"
    return _DATA_PARALLEL_GROUP


def get_tensor_model_parallel_rank():
    """Get tensor model parallel rank that the caller rank belongs to."""
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank(group=get_tensor_model_parallel_group())
    return 0


def get_tensor_model_parallel_group_rank():
    """Alias for get_tensor_model_parallel_rank"""
    return get_tensor_model_parallel_rank()


def get_data_parallel_group_rank():
    """Get data parallel rank that the caller rank belongs to."""
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank(group=get_data_parallel_group())
    return 0
