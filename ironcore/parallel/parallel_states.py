# Copyright (c) 2025-2026 Jaegeun Han
#
# SPDX-License-Identifier: Apache-2.0

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
    from ironcore.parallel.random import reset_tensor_parallel_rng_tracker

    reset_tensor_parallel_rng_tracker()
    # pylint: disable=global-statement

    global _TENSOR_MODEL_PARALLEL_WORLD_SIZE
    _TENSOR_MODEL_PARALLEL_WORLD_SIZE = tensor_model_parallel_size

    global _DATA_PARALLEL_WORLD_SIZE
    if dist.is_initialized():
        world_size = dist.get_world_size()
        _DATA_PARALLEL_WORLD_SIZE = world_size // tensor_model_parallel_size
    else:
        _DATA_PARALLEL_WORLD_SIZE = 1

    if not dist.is_initialized():
        return

    rank = dist.get_rank()
    world_size = dist.get_world_size()

    assert world_size % tensor_model_parallel_size == 0, (
        "world size must be divisible by tensor model parallel size"
    )

    timeout = timedelta(minutes=timeout_in_minutes)
    # Follow whatever the world group was built with. Hardcoding "nccl" here
    # ignored parallel.dist_backend, which parallel.py does honour — so setting
    # it to gloo produced a gloo world group with nccl TP/DP subgroups, and the
    # collectives that matter stayed on the backend the config asked to avoid.
    try:
        backend = dist.get_backend()
    except (RuntimeError, ValueError):
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
        [i * tensor_model_parallel_size + j for j in range(tensor_model_parallel_size)]
        for i in range(dp_world_size)
    ]
    for tp_group_id, ranks in enumerate(tp_ranks):  # pylint: disable=unused-variable
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
    dp_ranks = [[tp_group[i] for tp_group in tp_ranks] for i in range(tensor_model_parallel_size)]
    # create a new process group for data parallelism
    for dp_group_id, ranks in enumerate(dp_ranks):  # pylint: disable=unused-variable
        group = dist.new_group(
            ranks,
            timeout=timeout,
            backend=backend,
        )
        # group_desc=f"data parallel group ({dp_group_id})",
        if rank in ranks:
            _DATA_PARALLEL_GROUP = group


def destroy_model_parallel():
    """Clean up parallel groups and reset state.

    This function should be called before reinitializing with different
    parallel configuration (e.g., when switching from TP=1 to TP=2 in tests).

    Note: Process groups cannot be destroyed in PyTorch, so we only reset
    our references. The actual groups remain until process exit.
    """
    from ironcore.parallel.random import reset_tensor_parallel_rng_tracker

    reset_tensor_parallel_rng_tracker()

    global _TENSOR_MODEL_PARALLEL_WORLD_SIZE
    global _DATA_PARALLEL_WORLD_SIZE
    global _TENSOR_MODEL_PARALLEL_GROUP
    global _DATA_PARALLEL_GROUP

    _TENSOR_MODEL_PARALLEL_WORLD_SIZE = None
    _DATA_PARALLEL_WORLD_SIZE = None
    _TENSOR_MODEL_PARALLEL_GROUP = None
    _DATA_PARALLEL_GROUP = None


def is_model_parallel_initialized() -> bool:
    """Check if model parallel has been initialized."""
    return _TENSOR_MODEL_PARALLEL_WORLD_SIZE is not None


def get_data_parallel_world_size():
    """Get data parallel world size."""
    if _DATA_PARALLEL_WORLD_SIZE is None:
        raise RuntimeError("Data parallel not initialized. Call initialize_model_parallel() first.")
    return _DATA_PARALLEL_WORLD_SIZE


def get_tensor_model_parallel_world_size():
    """Get tensor model parallel world size."""
    if _TENSOR_MODEL_PARALLEL_WORLD_SIZE is None:
        raise RuntimeError(
            "Tensor model parallel not initialized. Call initialize_model_parallel() first."
        )
    return _TENSOR_MODEL_PARALLEL_WORLD_SIZE


def get_tensor_model_parallel_group() -> dist.ProcessGroup:
    """Get model parallel group that the caller rank belongs to."""
    if _TENSOR_MODEL_PARALLEL_GROUP is None:
        raise RuntimeError(
            "Tensor model parallel group not initialized. Call initialize_model_parallel() first."
        )
    return _TENSOR_MODEL_PARALLEL_GROUP


def get_data_parallel_group() -> dist.ProcessGroup:
    """Get data parallel group that the caller rank belongs to."""
    if _DATA_PARALLEL_GROUP is None:
        raise RuntimeError(
            "Data parallel group not initialized. Call initialize_model_parallel() first."
        )
    return _DATA_PARALLEL_GROUP


def get_tensor_model_parallel_rank() -> int:
    """Get tensor model parallel rank that the caller rank belongs to."""
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank(group=get_tensor_model_parallel_group())
    return 0


def get_tensor_model_parallel_group_rank() -> int:
    """Alias for get_tensor_model_parallel_rank"""
    return get_tensor_model_parallel_rank()


def get_data_parallel_group_rank() -> int:
    """Get data parallel rank that the caller rank belongs to."""
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank(group=get_data_parallel_group())
    return 0


def get_local_kv_group_info(num_kv_groups: int) -> tuple[int, int, int]:
    """
    Get local KV group info for tensor parallelism.

    This helper function calculates how many KV groups should be stored on
    the current TP rank when using tensor parallelism with GQA/MQA.

    Args:
        num_kv_groups: Total number of KV groups (global)

    Returns:
        tuple: (num_local_kv_groups, tp_rank, tp_size)
            - num_local_kv_groups: Number of KV groups on this rank
            - tp_rank: Current tensor parallel rank
            - tp_size: Tensor model parallel world size

    Raises:
        AssertionError: If tensor parallel is not initialized
        ValueError: If num_kv_groups is not divisible by tp_size

    Example:
        With TP=2 and 4 global KV groups:
        - Rank 0 gets 2 local KV groups (groups 0-1)
        - Rank 1 gets 2 local KV groups (groups 2-3)
    """
    tp_size = get_tensor_model_parallel_world_size()
    tp_rank = get_tensor_model_parallel_rank()

    if num_kv_groups % tp_size != 0:
        raise ValueError(
            f"num_kv_groups ({num_kv_groups}) must be divisible by "
            f"tensor_model_parallel_size ({tp_size}). "
            f"Consider using more KV groups or fewer TP ranks."
        )

    num_local_kv_groups = num_kv_groups // tp_size
    return num_local_kv_groups, tp_rank, tp_size


def ensure_tp_compatible(num_kv_groups: int) -> None:
    """
    Validate that the KV group configuration is compatible with TP.

    Args:
        num_kv_groups: Total number of KV groups (global)

    Raises:
        ValueError: If configuration is incompatible with TP
    """
    tp_size = get_tensor_model_parallel_world_size()

    if num_kv_groups < tp_size:
        raise ValueError(
            f"num_kv_groups ({num_kv_groups}) must be >= "
            f"tensor_model_parallel_size ({tp_size}). "
            f"Each TP rank needs at least one KV group. "
            f"Consider reducing TP size or increasing KV groups."
        )

    if num_kv_groups % tp_size != 0:
        raise ValueError(
            f"num_kv_groups ({num_kv_groups}) must be divisible by "
            f"tensor_model_parallel_size ({tp_size})."
        )
