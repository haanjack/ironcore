# Copyright (c) 2025-2026 Jaegeun Han
#
# SPDX-License-Identifier: Apache-2.0

"""Gradient norm computation for distributed training."""

from collections.abc import Iterable
from typing import Union

import torch
from torch import distributed as dist

from ironcore.parallel import parallel_states
from ironcore.parallel.expert_parallel.parallel_states import (
    get_expert_model_parallel_group,
    get_expert_model_parallel_world_size,
)


def clip_grad_norm(
    parameters: Union[torch.Tensor, Iterable[torch.Tensor]],
    max_norm: float,
    norm_type: float = 2.0,
) -> torch.Tensor:
    """
    Clips gradient norm of an iterable of parameters across distributed training.

    Supports:
    - Tensor Parallelism (TP): parameters sharded across TP ranks
    - Expert Parallelism (EP): expert parameters sharded across EP ranks
    - Data Parallelism (DP): replicated parameters across DP ranks
    (FSDP is handled separately in the trainer)

    Args:
        parameters: Iterable of Tensors or a single Tensor to be normalized.
        max_norm: Maximum norm of the gradients.
        norm_type: Type of the used p-norm. Use torch.inf for infinity norm.

    Returns:
        total_norm: The computed total norm of the gradients.
    """
    from torch import inf

    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]

    # Filter parameters that have gradients
    grads = [p.grad for p in parameters if p.grad is not None]

    max_norm = float(max_norm)
    norm_type = float(norm_type)

    if len(grads) == 0:
        return torch.tensor(0.0, device=torch.cuda.current_device())

    device = grads[0].device

    # --- Step 1: Calculate Local Norm ---
    if norm_type == inf:
        # Calculate local max absolute value
        total_norm = max(g.detach().abs().max() for g in grads)
        total_norm = torch.tensor(float(total_norm), device=device)
    else:
        # Calculate local sum of powers: sum(||g||^p)
        total_norm_pow = (
            torch.norm(torch.stack([torch.norm(g.detach(), norm_type) for g in grads]), norm_type)
            ** norm_type
        )

    # --- Step 2: Communication across Tensor Parallel (TP) Group ---
    # Since non-expert parameters are sharded across TP ranks, we MUST sum/max them.
    tp_size = parallel_states.get_tensor_model_parallel_world_size()
    if tp_size > 1:
        tp_group = parallel_states.get_tensor_model_parallel_group()
        dist.all_reduce(
            total_norm if norm_type == inf else total_norm_pow,
            op=dist.ReduceOp.MAX if norm_type == inf else dist.ReduceOp.SUM,
            group=tp_group,
        )

    # --- Step 3: Communication across Expert Parallel (EP) Group ---
    # If MoE is enabled, expert parameters are sharded across EP ranks.
    # Non-expert parameters are replicated across EP, so we don't double-count them.
    try:
        ep_group = get_expert_model_parallel_group()
        if ep_group is not None and get_expert_model_parallel_world_size() > 1:
            dist.all_reduce(
                total_norm if norm_type == inf else total_norm_pow,
                op=dist.ReduceOp.MAX if norm_type == inf else dist.ReduceOp.SUM,
                group=ep_group,
            )
    except (ImportError, AttributeError, RuntimeError):
        # MoE not enabled or expert parallel not initialized
        pass

    # --- Step 4: Communication across Data Parallel (DP) Group ---
    # For DDP, gradients are already averaged. Summing them again would
    # scale the norm by DP_size, so we must average the power sum.
    dp_size = parallel_states.get_data_parallel_world_size()
    if dist.is_initialized() and dp_size > 1:
        dp_group = parallel_states.get_data_parallel_group()

        if norm_type == inf:
            # Sync max value to ensure bit-level consistency across all DP ranks
            dist.all_reduce(total_norm, op=dist.ReduceOp.MAX, group=dp_group)
        else:
            # Average the power sum across DP ranks to maintain mathematical correctness
            dist.all_reduce(total_norm_pow, op=dist.ReduceOp.SUM, group=dp_group)
            total_norm_pow /= dp_size

    # --- Step 5: Finalize Total Norm ---
    if norm_type != inf:
        total_norm = total_norm_pow ** (1.0 / norm_type)

    # --- Step 6: Apply Clipping ---
    clip_coef = max_norm / (total_norm + 1e-6)
    if clip_coef < 1.0:
        for g in grads:
            g.detach().mul_(clip_coef)

    return total_norm
