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
    parameters = [p for p in parameters if p.grad is not None]
    if len(parameters) == 0:
        return torch.tensor(0.0, device=torch.cuda.current_device())

    grads = [p.grad for p in parameters]
    max_norm = float(max_norm)
    norm_type = float(norm_type)
    device = grads[0].device

    # Separate expert and non-expert gradients
    expert_grads = [p.grad for p in parameters if getattr(p, "is_expert", False)]
    non_expert_grads = [p.grad for p in parameters if not getattr(p, "is_expert", False)]

    # --- Step 1: Calculate Local Norms ---
    if norm_type == inf:
        local_expert = max(g.detach().abs().max() for g in expert_grads) if expert_grads else 0.0
        local_non_expert = max(g.detach().abs().max() for g in non_expert_grads) if non_expert_grads else 0.0
        
        # Use tensors for collective communication
        norms = torch.tensor([float(local_expert), float(local_non_expert)], device=device)
        
        # --- Step 2: TP Reduction ---
        tp_size = parallel_states.get_tensor_model_parallel_world_size()
        if tp_size > 1:
            dist.all_reduce(norms, op=dist.ReduceOp.MAX, group=parallel_states.get_tensor_model_parallel_group())
            
        # --- Step 3: EP Reduction ---
        ep_size = get_expert_model_parallel_world_size()
        if ep_size > 1:
            # Expert parameters are sharded across EP, non-expert are replicated
            dist.all_reduce(norms[0], op=dist.ReduceOp.MAX, group=get_expert_model_parallel_group())
            
        # --- Step 4: DP Reduction ---
        dp_size = parallel_states.get_data_parallel_world_size()
        if dp_size > 1:
            dist.all_reduce(norms, op=dist.ReduceOp.MAX, group=parallel_states.get_data_parallel_group())
            
        total_norm = norms.max()
        
    else:
        # Calculate local power sums: sum(||g||^p)
        local_expert_pow = torch.stack([g.detach().norm(norm_type)**norm_type for g in expert_grads]).sum() if expert_grads else torch.tensor(0.0, device=device)
        local_non_expert_pow = torch.stack([g.detach().norm(norm_type)**norm_type for g in non_expert_grads]).sum() if non_expert_grads else torch.tensor(0.0, device=device)
        
        # --- Step 2: TP Reduction ---
        tp_size = parallel_states.get_tensor_model_parallel_world_size()
        if tp_size > 1:
            tp_group = parallel_states.get_tensor_model_parallel_group()
            dist.all_reduce(local_expert_pow, op=dist.ReduceOp.SUM, group=tp_group)
            dist.all_reduce(local_non_expert_pow, op=dist.ReduceOp.SUM, group=tp_group)
            
        # --- Step 3: EP Reduction ---
        ep_size = get_expert_model_parallel_world_size()
        if ep_size > 1:
            # Expert parameters are sharded across EP ranks, so we SUM.
            # Non-expert parameters are REPLICATED across EP ranks, so we stay with current value.
            dist.all_reduce(local_expert_pow, op=dist.ReduceOp.SUM, group=get_expert_model_parallel_group())
            
        # --- Step 4: DP Reduction (for DDP averaging) ---
        dp_size = parallel_states.get_data_parallel_world_size()
        if dist.is_initialized() and dp_size > 1:
            dp_group = parallel_states.get_data_parallel_group()
            # Gradients in DDP are averaged, so norm^p is averaged.
            dist.all_reduce(local_expert_pow, op=dist.ReduceOp.SUM, group=dp_group)
            dist.all_reduce(local_non_expert_pow, op=dist.ReduceOp.SUM, group=dp_group)
            local_expert_pow /= dp_size
            local_non_expert_pow /= dp_size
            
        total_norm_pow = local_expert_pow + local_non_expert_pow
        total_norm = total_norm_pow ** (1.0 / norm_type)

    # --- Step 5: Apply Clipping ---
    clip_coef = max_norm / (total_norm + 1e-6)
    if clip_coef < 1.0:
        for g in grads:
            g.detach().mul_(clip_coef)

    return total_norm
