# Copyright (c) 2025-2026 Jaegeun Han
# SPDX-License-Identifier: Apache-2.0

"""Padding-free expert execution with bounded native CUDA grouped GEMM."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from ironcore.parallel.random import checkpoint_with_tensor_parallel_rng
from ironcore.parallel.tensor_parallel import comm

from .execution_groups import plan_execution_groups


def _grouped_mm(
    inputs: torch.Tensor, weights: torch.Tensor, offsets: torch.Tensor, counts: tuple[int, ...]
) -> torch.Tensor:
    if inputs.is_cuda:
        kernel = getattr(F, "grouped_mm", None)
        if kernel is None:
            raise RuntimeError("Grouped experts require torch.nn.functional.grouped_mm on CUDA")
        return kernel(inputs, weights, offs=offsets)
    # Independent FP32 CPU reference, not a claim of grouped CPU acceleration.
    return torch.cat(
        [part @ weight for part, weight in zip(inputs.split(counts), weights, strict=True)]
    )


def grouped_experts(
    x: torch.Tensor,
    indices: torch.Tensor,
    weights: torch.Tensor,
    experts: torch.nn.ModuleList,
) -> torch.Tensor:
    """Route once, fill row-budget groups, and checkpoint bounded jagged GEMMs.

    Only the execution group's active weights are packed inside its checkpoint,
    preserving grad=None for idle experts without retaining layer-wide copies.
    Grouped GEMM does not participate in CUDA
    autocast on all supported versions, so compute dtype is selected explicitly.
    Expanded up/gate/activation intermediates live within one execution group.
    """
    if experts[0].config.model.moe.blockwise_backend != "torch":
        from .scheduled import scheduled_experts

        return scheduled_experts(x, indices, weights, experts, padded=False)
    config = experts[0].config.model.moe
    hidden = x.reshape(-1, x.size(-1))
    topk = indices.size(-1)
    ids = indices.reshape(-1)
    if ids.numel() == 0:
        return x * 0
    order = ids.argsort(stable=True)
    sizes = torch.bincount(ids, minlength=len(experts)).tolist()
    groups = plan_execution_groups(sizes, config.grouped_token_budget)
    active = [i for i, count in enumerate(sizes) if count]
    compute_dtype = (
        torch.get_autocast_dtype(x.device.type)
        if torch.is_autocast_enabled(x.device.type)
        else x.dtype
    )
    down_bias = (
        torch.stack([experts[i].down_proj.bias for i in active])
        if experts[0].down_proj.bias is not None
        else None
    )
    # A single compact gather replaces repeated expert-capacity padding.
    routed = comm.copy_inputs_to_model_parallel_workers(hidden)[order // topk].to(compute_dtype)
    endpoints = torch.tensor(
        [v for group in groups for v in group.offsets], dtype=torch.int32, device=x.device
    )
    outputs = []
    offset_start = 0
    for group in groups:
        offsets = endpoints[offset_start : offset_start + len(group.experts)]
        offset_start += len(group.experts)

        # Bind group metadata: checkpoint closures outlive this loop iteration.
        def project(tokens, owners=group.experts, offsets=offsets, counts=group.counts):
            up = torch.stack([experts[i].up_proj.weight for i in owners]).to(compute_dtype)
            down = torch.stack([experts[i].down_proj.weight for i in owners]).to(compute_dtype)
            projected = _grouped_mm(tokens, up, offsets, counts)
            if experts[0].up_proj.bias is not None:
                up_bias = torch.stack([experts[i].up_proj.bias for i in owners])
                owners = torch.repeat_interleave(
                    torch.arange(len(owners), device=x.device),
                    torch.tensor(counts, device=x.device),
                    output_size=tokens.size(0),
                )
                projected = projected + up_bias[owners]
            activated = experts[0].activation(projected).to(compute_dtype)
            result = _grouped_mm(activated, down, offsets, counts)
            return result

        tokens = routed[group.start : group.start + group.tokens]
        if experts[0].training and torch.is_grad_enabled():
            result = checkpoint_with_tensor_parallel_rng(project, tokens, use_reentrant=False)
        else:
            result = project(tokens)
        outputs.append(result)
    sorted_outputs = torch.cat(outputs) if len(outputs) > 1 else outputs[0]
    original = sorted_outputs[order.argsort()].view(-1, topk, hidden.size(-1))
    parallel_weights = comm.copy_inputs_to_model_parallel_workers(weights)
    result = (original * parallel_weights.reshape(-1, topk, 1)).sum(1)
    result = comm.reduce_inputs_from_model_parallel_workers(result)
    if down_bias is not None:
        # Map original global expert ids to compact active weights without
        # creating optimizer gradients for globally idle experts.
        lookup = torch.zeros(len(experts), dtype=torch.long, device=x.device)
        lookup[torch.tensor(active, device=x.device)] = torch.arange(len(active), device=x.device)
        selected_bias = down_bias[lookup[indices.reshape(-1, topk)]]
        result = result + (selected_bias * weights.reshape(-1, topk, 1)).sum(1)
    return result.view_as(x)
