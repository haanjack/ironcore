# Copyright (c) 2025-2026 Jaegeun Han
# SPDX-License-Identifier: Apache-2.0

"""Padding-free expert execution with bounded native CUDA grouped GEMM."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from ironcore.parallel.random import checkpoint_with_tensor_parallel_rng
from ironcore.parallel.tensor_parallel import comm

from .execution_groups import plan_execution_groups
from .lora import expert_parameters


def _grouped_mm(
    inputs: torch.Tensor, weights: torch.Tensor, offsets: torch.Tensor, counts: tuple[int, ...]
) -> torch.Tensor:
    if inputs.is_cuda:
        kernel = getattr(F, "grouped_mm", None)
        if kernel is None:
            raise RuntimeError("Grouped experts require torch.nn.functional.grouped_mm on CUDA")
        # CUDA grouped GEMM requires 16-byte aligned row strides. Low LoRA
        # ranks (and tiny TP shards) can violate that even with contiguous
        # tensors. Pad only the compute operands, preserving adapter shapes.
        alignment = 16 // inputs.element_size()
        inner, output = weights.shape[-2:]
        inner_pad, output_pad = -inner % alignment, -output % alignment
        if inner_pad:
            inputs = F.pad(inputs, (0, inner_pad))
        if inner_pad or output_pad:
            weights = F.pad(weights, (0, output_pad, 0, inner_pad))
        return kernel(inputs, weights, offs=offsets)[..., :output]
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
    if experts[0].config.model.moe.expert_accumulation_precision == "model":
        from .model_accumulation import model_dtype_experts

        return model_dtype_experts(x, indices, weights, experts)
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
    from .scheduled import _Block, _project

    parameters, adapters, stride = expert_parameters(experts, active)
    # Group planning uses global ids; _project indexes the compact active list.
    compact = {owner: i for i, owner in enumerate(active)}
    outputs = []
    for group in groups:
        block = _Block(tuple(compact[i] for i in group.experts), group.start, group.counts)

        def project(tokens, block=block):
            return _project(
                tokens,
                parameters,
                block,
                experts[0].activation,
                experts[0].up_proj.bias is not None,
                False,
                compute_dtype,
                adapters,
                stride,
            )

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
