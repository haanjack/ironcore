# Copyright (c) 2025-2026 Jaegeun Han
#
# SPDX-License-Identifier: Apache-2.0

"""Expert parallel process group management.

This module manages process groups for Expert Parallelism (EP), which distributes
routed experts across multiple GPUs. Each GPU holds a subset of experts.

Layout Example (EP=2, TP=2, World=4):
- EP Group 0: Ranks [0, 1] hold experts 0-31 (TP-sharded)
- EP Group 1: Ranks [2, 3] hold experts 32-63 (TP-sharded)

Within each EP group, experts are further sharded using Tensor Parallelism.
"""

from datetime import timedelta

import torch
from torch import distributed as dist

# Expert parallel state
_EXPERT_MODEL_PARALLEL_GROUP: dist.ProcessGroup | None = None
_EXPERT_MODEL_PARALLEL_WORLD_SIZE: int = 1
_EXPERT_MODEL_PARALLEL_RANK: int = 0

# Tensor parallel within expert
_EXPERT_TENSOR_PARALLEL_GROUP_WITHIN_EP: dist.ProcessGroup | None = None


def initialize_expert_parallel(
    expert_model_parallel_size: int,
    tensor_model_parallel_size: int,
    timeout_in_minutes: float = 10.0,
) -> None:
    """Initialize process groups for Expert Parallelism.

    Creates two types of process groups:
    1. Expert Model Parallel (EP): Ranks that hold different expert subsets
    2. Tensor Parallel within EP: Ranks within same EP group for TP sharding

    Layout:
        World is divided into EP groups, each containing TP ranks.
        - EP rank determines which expert subset this rank holds
        - TP rank determines the shard within each expert

    Example (EP=2, TP=2, World=4):
        - EP Group 0: [0, 1] -> experts 0-31
        - EP Group 1: [2, 3] -> experts 32-63

        TP within EP:
        - [0]: TP rank 0 within EP group 0
        - [1]: TP rank 1 within EP group 0
        - [2]: TP rank 0 within EP group 1
        - [3]: TP rank 1 within EP group 1

    Args:
        expert_model_parallel_size: Number of EP groups (how many expert subsets)
        tensor_model_parallel_size: TP size within each expert
        timeout_in_minutes: Timeout for process group operations
    """
    # pylint: disable=global-statement
    global _EXPERT_MODEL_PARALLEL_GROUP
    global _EXPERT_MODEL_PARALLEL_WORLD_SIZE
    global _EXPERT_MODEL_PARALLEL_RANK
    global _EXPERT_TENSOR_PARALLEL_GROUP_WITHIN_EP

    _EXPERT_MODEL_PARALLEL_WORLD_SIZE = expert_model_parallel_size

    if not dist.is_initialized():
        _EXPERT_MODEL_PARALLEL_RANK = 0
        return

    rank = dist.get_rank()
    world_size = dist.get_world_size()

    # Validate world size
    total_parallel_size = expert_model_parallel_size * tensor_model_parallel_size
    if world_size % total_parallel_size != 0:
        raise ValueError(
            f"World size ({world_size}) must be divisible by "
            f"expert_model_parallel_size * tensor_model_parallel_size ({total_parallel_size})"
        )

    dp_world_size = world_size // total_parallel_size
    timeout = timedelta(minutes=timeout_in_minutes)
    backend = "nccl" if torch.cuda.is_available() else "gloo"

    # Calculate EP and TP ranks from global rank
    # Layout: [DP][EP][TP]
    # rank = dp_idx * (ep_size * tp_size) + ep_idx * tp_size + tp_idx
    rank_in_dp_group = rank % (expert_model_parallel_size * tensor_model_parallel_size)
    _EXPERT_MODEL_PARALLEL_RANK = rank_in_dp_group // tensor_model_parallel_size

    # Initialize EP groups: ranks that hold different expert subsets
    # EP groups are formed by ranks with the same TP index across different EP positions
    # For EP=2, TP=2: EP groups are [[0,2], [1,3]] within each DP group
    ep_ranks = []
    for dp_idx in range(dp_world_size):
        for tp_idx in range(tensor_model_parallel_size):
            group_ranks = [
                dp_idx * total_parallel_size + ep_idx * tensor_model_parallel_size + tp_idx
                for ep_idx in range(expert_model_parallel_size)
            ]
            ep_ranks.append(group_ranks)

    for ranks in ep_ranks:
        group = dist.new_group(ranks, timeout=timeout, backend=backend)
        if rank in ranks:
            _EXPERT_MODEL_PARALLEL_GROUP = group

    # Initialize TP groups within EP: ranks within same EP that shard the same experts
    # For EP=2, TP=2: TP groups are [[0,1], [2,3]] within each DP group
    if tensor_model_parallel_size > 1:
        tp_within_ep_ranks = []
        for dp_idx in range(dp_world_size):
            for ep_idx in range(expert_model_parallel_size):
                group_ranks = [
                    dp_idx * total_parallel_size + ep_idx * tensor_model_parallel_size + tp_idx
                    for tp_idx in range(tensor_model_parallel_size)
                ]
                tp_within_ep_ranks.append(group_ranks)

        for ranks in tp_within_ep_ranks:
            group = dist.new_group(ranks, timeout=timeout, backend=backend)
            if rank in ranks:
                _EXPERT_TENSOR_PARALLEL_GROUP_WITHIN_EP = group


def destroy_expert_parallel() -> None:
    """Destroy expert parallel process groups."""
    # pylint: disable=global-statement
    global _EXPERT_MODEL_PARALLEL_GROUP
    global _EXPERT_MODEL_PARALLEL_WORLD_SIZE
    global _EXPERT_MODEL_PARALLEL_RANK
    global _EXPERT_TENSOR_PARALLEL_GROUP_WITHIN_EP

    _EXPERT_MODEL_PARALLEL_GROUP = None
    _EXPERT_MODEL_PARALLEL_WORLD_SIZE = 1
    _EXPERT_MODEL_PARALLEL_RANK = 0
    _EXPERT_TENSOR_PARALLEL_GROUP_WITHIN_EP = None


def get_expert_model_parallel_group() -> dist.ProcessGroup | None:
    """Get the expert model parallel process group.

    This group contains all ranks that hold different expert subsets.
    Used for all-to-all communication during token routing.
    """
    return _EXPERT_MODEL_PARALLEL_GROUP


def get_expert_model_parallel_world_size() -> int:
    """Get the expert model parallel world size (number of expert subsets)."""
    return _EXPERT_MODEL_PARALLEL_WORLD_SIZE


def get_expert_model_parallel_rank() -> int:
    """Get the expert model parallel rank (which expert subset this rank holds)."""
    return _EXPERT_MODEL_PARALLEL_RANK


def get_expert_tensor_parallel_group_within_ep() -> dist.ProcessGroup | None:
    """Get the tensor parallel group within the same EP rank.

    This group contains ranks that shard the same expert subset.
    """
    return _EXPERT_TENSOR_PARALLEL_GROUP_WITHIN_EP


def get_local_expert_indices(
    num_routed_experts: int,
    expert_model_parallel_size: int,
) -> tuple[int, int]:
    """Get the range of expert indices owned by this rank.

    Args:
        num_routed_experts: Total number of routed experts
        expert_model_parallel_size: Number of EP groups

    Returns:
        Tuple of (start_index, end_index) for local experts
    """
    if expert_model_parallel_size == 1:
        return 0, num_routed_experts

    ep_rank = get_expert_model_parallel_rank()
    experts_per_rank = num_routed_experts // expert_model_parallel_size

    # Ensure even division
    if num_routed_experts % expert_model_parallel_size != 0:
        raise ValueError(
            f"num_routed_experts ({num_routed_experts}) must be divisible by "
            f"expert_model_parallel_size ({expert_model_parallel_size})"
        )

    start_idx = ep_rank * experts_per_rank
    end_idx = start_idx + experts_per_rank

    return start_idx, end_idx
