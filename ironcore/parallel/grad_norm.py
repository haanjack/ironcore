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
    params_with_grad = [p for p in parameters if p.grad is not None]
    if len(params_with_grad) == 0:
        return torch.tensor(0.0)

    grads = [p.grad for p in params_with_grad]
    max_norm = float(max_norm)
    norm_type = float(norm_type)
    device = grads[0].device

    # Helper to calculate local power sum for a group of parameters
    def get_local_pow_sum(params):
        sharded_pow = torch.tensor(0.0, device=device)
        replicated_pow = torch.tensor(0.0, device=device)
        for p in params:
            p_pow = p.grad.detach().norm(norm_type) ** norm_type
            if getattr(p, "is_tp_sharded", False):
                sharded_pow += p_pow
            else:
                replicated_pow += p_pow
        return sharded_pow, replicated_pow

    expert_params = [p for p in params_with_grad if getattr(p, "is_expert", False)]
    non_expert_params = [p for p in params_with_grad if not getattr(p, "is_expert", False)]

    # --- Step 1: Calculate Local Norms ---
    if norm_type == inf:
        local_expert = (
            max(p.grad.detach().abs().max() for p in expert_params) if expert_params else 0.0
        )
        local_non_expert = (
            max(p.grad.detach().abs().max() for p in non_expert_params) if non_expert_params else 0.0
        )

        # Use tensors for collective communication
        norms = torch.tensor([float(local_expert), float(local_non_expert)], device=device)

        # --- Step 2: TP Reduction ---
        tp_size = parallel_states.get_tensor_model_parallel_world_size()
        if tp_size > 1:
            # For infinity norm, MAX reduction is correct for both sharded and replicated
            dist.all_reduce(
                norms, op=dist.ReduceOp.MAX, group=parallel_states.get_tensor_model_parallel_group()
            )

        # --- Step 3: EP Reduction ---
        ep_size = get_expert_model_parallel_world_size()
        if ep_size > 1:
            # Expert parameters are sharded across EP, non-expert are replicated
            dist.all_reduce(norms[0], op=dist.ReduceOp.MAX, group=get_expert_model_parallel_group())

        # --- Step 4: DP Reduction ---
        dp_size = parallel_states.get_data_parallel_world_size()
        if dp_size > 1:
            dist.all_reduce(
                norms, op=dist.ReduceOp.MAX, group=parallel_states.get_data_parallel_group()
            )

        total_norm = norms.max()

    else:
        # Calculate local power sums
        exp_sharded_pow, exp_repl_pow = get_local_pow_sum(expert_params)
        non_exp_sharded_pow, non_exp_repl_pow = get_local_pow_sum(non_expert_params)

        # --- Step 2: TP Reduction ---
        tp_size = parallel_states.get_tensor_model_parallel_world_size()
        if tp_size > 1:
            tp_group = parallel_states.get_tensor_model_parallel_group()
            # Combine sharded and replicated for all-reduce
            combined = torch.stack(
                [exp_sharded_pow, exp_repl_pow, non_exp_sharded_pow, non_exp_repl_pow]
            )
            dist.all_reduce(combined, op=dist.ReduceOp.SUM, group=tp_group)

            # For replicated parameters, SUM across TP over-counts, so we divide
            exp_sharded_pow, exp_repl_pow, non_exp_sharded_pow, non_exp_repl_pow = combined
            exp_repl_pow /= tp_size
            non_exp_repl_pow /= tp_size

        local_expert_pow = exp_sharded_pow + exp_repl_pow
        local_non_expert_pow = non_exp_sharded_pow + non_exp_repl_pow

        # --- Step 3: EP Reduction ---
        ep_size = get_expert_model_parallel_world_size()
        if ep_size > 1:
            # Expert parameters are sharded across EP ranks, so we SUM.
            # Non-expert parameters are REPLICATED across EP ranks, so we stay with current value.
            dist.all_reduce(
                local_expert_pow, op=dist.ReduceOp.SUM, group=get_expert_model_parallel_group()
            )

        # --- Step 4: DP Reduction (for DDP averaging) ---
        dp_size = parallel_states.get_data_parallel_world_size()
        if dist.is_initialized() and dp_size > 1:
            dp_group = parallel_states.get_data_parallel_group()
            # Combine for single all-reduce
            combined = torch.stack([local_expert_pow, local_non_expert_pow])
            dist.all_reduce(combined, op=dist.ReduceOp.SUM, group=dp_group)
            local_expert_pow, local_non_expert_pow = combined / dp_size

        total_norm_pow = local_expert_pow + local_non_expert_pow
        total_norm = total_norm_pow ** (1.0 / norm_type)

    # --- Step 5: Apply Clipping ---
    clip_coef = max_norm / (total_norm + 1e-6)
    if clip_coef < 1.0:
        # Use a set to avoid double-clipping shared gradients
        for g in {p.grad for p in params_with_grad}:
            g.detach().mul_(clip_coef)

    return total_norm
