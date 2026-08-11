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
            max(p.grad.detach().abs().max() for p in non_expert_params)
            if non_expert_params
            else 0.0
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
            combined = torch.stack([local_expert_pow, local_non_expert_pow])
            # NCCL requires CUDA tensors; move to GPU temporarily if needed
            combined_device = combined.device
            if combined_device.type == "cpu":
                combined = combined.cuda()
            dist.all_reduce(combined, op=dist.ReduceOp.SUM, group=dp_group)
            if combined_device.type == "cpu":
                combined = combined.to("cpu")
            # With ZeRO-3, gradients are already partitioned (distinct across ranks),
            # so SUM gives the correct total without dividing by dp_size.
            # With standard DDP, gradients are replicated, so divide by dp_size.
            grads_are_sharded = any(
                p.grad.numel() < p.numel() for p in params_with_grad if p.grad is not None
            )
            if not grads_are_sharded:
                local_expert_pow, local_non_expert_pow = combined / dp_size
            else:
                local_expert_pow, local_non_expert_pow = combined[0], combined[1]

        total_norm_pow = local_expert_pow + local_non_expert_pow
        total_norm = total_norm_pow ** (1.0 / norm_type)

    # --- Step 5: Apply Clipping ---
    clip_coef = max_norm / (total_norm + 1e-6)
    if clip_coef < 1.0:
        # Use a set to avoid double-clipping shared gradients
        for g in {p.grad for p in params_with_grad}:
            g.detach().mul_(clip_coef.to(g.device))

    return total_norm


def compute_param_norm(
    parameters: Union[torch.Tensor, Iterable[torch.Tensor]],
    is_fsdp: bool = False,
) -> float:
    """
    Compute the global L2 parameter norm across distributed training.

    Shared by LanguageModelTrainer and GRPOTrainer so both report the same,
    correctly-TP-aware value (a prior duplicate in GRPOTrainer summed replicated
    params across the TP group without dividing by tp_size, over-counting them).

    Supports:
    - Tensor Parallelism (TP): parameters sharded across TP ranks (SUM, then
      divide replicated-param contributions by tp_size since they were
      counted once per rank)
    - Expert Parallelism (EP): expert parameters sharded across EP ranks (SUM)
    - Data Parallelism (DP): replicated parameters across DP ranks (SUM then
      average), or FSDP-sharded parameters across the DP group (SUM)

    Args:
        parameters: Iterable of Tensors (or a single Tensor) to compute the norm over.
        is_fsdp: Whether `parameters` come from an FSDP-wrapped model (parameters
            are sharded across the data-parallel group in that case).

    Returns:
        The computed global parameter norm as a Python float.
    """
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]

    params = [p for p in parameters if p.data is not None]
    if len(params) == 0:
        return 0.0

    device = params[0].data.device

    expert_params = [p for p in params if getattr(p, "is_expert", False)]
    non_expert_params = [p for p in params if not getattr(p, "is_expert", False)]

    expert_sharded = [p for p in expert_params if getattr(p, "is_tp_sharded", False)]
    expert_repl = [p for p in expert_params if not getattr(p, "is_tp_sharded", False)]
    non_expert_sharded = [p for p in non_expert_params if getattr(p, "is_tp_sharded", False)]
    non_expert_repl = [p for p in non_expert_params if not getattr(p, "is_tp_sharded", False)]

    def _norm_sq(params_subset):
        if not params_subset:
            return torch.zeros((), device=device)
        return torch.stack([p.data.norm() ** 2 for p in params_subset]).sum()

    expert_sharded_norm_sq = _norm_sq(expert_sharded)
    expert_repl_norm_sq = _norm_sq(expert_repl)
    non_expert_sharded_norm_sq = _norm_sq(non_expert_sharded)
    non_expert_repl_norm_sq = _norm_sq(non_expert_repl)

    # Step 1: FSDP reduction (parameters sharded across the DP group)
    if is_fsdp:
        combined = torch.stack([expert_sharded_norm_sq, non_expert_sharded_norm_sq])
        dist.all_reduce(
            combined, op=dist.ReduceOp.SUM, group=parallel_states.get_data_parallel_group()
        )
        expert_sharded_norm_sq, non_expert_sharded_norm_sq = combined

    # Step 2: Tensor parallelism reduction
    tp_size = parallel_states.get_tensor_model_parallel_world_size()
    if tp_size > 1:
        tp_group = parallel_states.get_tensor_model_parallel_group()
        combined = torch.stack(
            [
                expert_sharded_norm_sq,
                expert_repl_norm_sq,
                non_expert_sharded_norm_sq,
                non_expert_repl_norm_sq,
            ]
        )
        dist.all_reduce(combined, op=dist.ReduceOp.SUM, group=tp_group)

        # Replicated parameters were summed once per TP rank, so SUM over-counts them.
        (
            expert_sharded_norm_sq,
            expert_repl_norm_sq,
            non_expert_sharded_norm_sq,
            non_expert_repl_norm_sq,
        ) = combined
        expert_repl_norm_sq = expert_repl_norm_sq / tp_size
        non_expert_repl_norm_sq = non_expert_repl_norm_sq / tp_size

    expert_norm_sq = expert_sharded_norm_sq + expert_repl_norm_sq
    non_expert_norm_sq = non_expert_sharded_norm_sq + non_expert_repl_norm_sq

    # Step 3: Expert parallelism reduction (expert parameters sharded across EP group)
    ep_group = get_expert_model_parallel_group()
    if ep_group is not None and get_expert_model_parallel_world_size() > 1:
        dist.all_reduce(expert_norm_sq, op=dist.ReduceOp.SUM, group=ep_group)

    param_norm_sq = expert_norm_sq + non_expert_norm_sq

    # Step 4: DP reduction (replicated parameters in non-FSDP DP must be averaged,
    # not summed, since every DP rank holds an identical copy)
    dp_size = parallel_states.get_data_parallel_world_size()
    if dist.is_initialized() and not is_fsdp and dp_size > 1:
        dist.all_reduce(
            param_norm_sq, op=dist.ReduceOp.SUM, group=parallel_states.get_data_parallel_group()
        )
        param_norm_sq = param_norm_sq / dp_size

    return param_norm_sq.item() ** 0.5
