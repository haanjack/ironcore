# Copyright (c) 2025-2026 Jaegeun Han
# SPDX-License-Identifier: Apache-2.0
"""Bounded grouped GEMM with expert-ordered, model-dtype mixture accumulation.

Weight products round before addition, matching the HF Gemma expert loop.
CUDA pointwise products write directly to model-dtype buffers; FP32 arithmetic
inside registers does not require an activation-sized FP32 allocation.
"""

import torch
from torch.autograd.function import once_differentiable

from ironcore.parallel.random import (
    snapshot_tensor_parallel_rng_tracker,
    tensor_parallel_rng_rewound_to,
)
from ironcore.parallel.tensor_parallel import comm

from .execution_groups import plan_execution_groups
from .lora import expert_parameters
from .scheduled import _Block, _pack, _project, _scatter


def _segments(counts, reverse=False):
    segments = []
    start = 0
    for count in counts:
        segments.append(slice(start, start + count))
        start += count
    return reversed(segments) if reverse else segments


def _accumulate(
    destination, values, token_ids, counts, *, weights=None, fused=False, reverse=False
):
    # Groups can contain several experts, but each launch handles one expert.
    # Unique top-k choices mean rows within that expert never share a token.
    # Stream order preserves ascending expert order, including split experts.
    for segment in _segments(counts, reverse):
        projected, ids = values[segment], token_ids[segment]
        if weights is None:
            _scatter(destination, projected, ids, fused, unique=True)
        elif fused:
            from .triton_routing import weighted_scatter

            weighted_scatter(
                destination,
                projected,
                weights[segment],
                ids,
                unique=True,
                round_to_destination=True,
            )
        else:
            contribution = torch.empty_like(projected, dtype=destination.dtype)
            # Custom forward is no-grad: out= is safe and prevents the FP32
            # weight operand from allocating a FP32 product tensor on CUDA.
            torch.mul(projected, weights[segment, None], out=contribution)
            destination.index_add_(0, ids, contribution)


def _mixture_backward(projected, weights, incoming):
    grad_projected = torch.empty_like(projected)
    torch.mul(incoming, weights[:, None], out=grad_projected)
    if projected.is_cuda:
        from .triton_routing import mixture_weight_gradient

        grad_weights = mixture_weight_gradient(projected, incoming, weights.dtype)
    else:
        # CPU numerical oracle. CUDA performs this product/reduction in
        # registers and writes only one scalar per routing assignment.
        dtype = torch.promote_types(projected.dtype, weights.dtype)
        grad_weights = (projected.to(dtype) * incoming.to(dtype)).sum(-1).to(weights.dtype)
    return grad_projected, grad_weights


class _ModelDtypeExperts(torch.autograd.Function):
    @staticmethod
    def forward(ctx, hidden, weights, order, prefixes, counts, metadata, *parameters):
        blocks, activation, fused, unique, topk, adapters, stride = metadata
        device_type = hidden.device.type
        enabled = torch.is_autocast_enabled(device_type)
        compute_dtype = torch.get_autocast_dtype(device_type) if enabled else hidden.dtype
        result = torch.zeros_like(hidden)
        ctx.save_for_backward(hidden, weights, order, prefixes, counts, *parameters)
        ctx.execution_metadata = metadata
        ctx.autocast = (device_type, enabled, compute_dtype)
        ctx.parameter_needs_grad = tuple(p.requires_grad for p in parameters)
        ctx.rng_states = []
        for block in blocks:
            ctx.rng_states.append(
                snapshot_tensor_parallel_rng_tracker() if any(a.dropout for a in adapters) else {}
            )
            tokens, selected, _, ids = _pack(
                hidden, weights, order, prefixes, counts, block, topk, fused
            )
            projected = _project(
                tokens, parameters, block, activation, False, False, compute_dtype, adapters, stride
            )
            # TP must combine the expert output before rounding its weighted
            # contribution. Reducing an already rounded mixture changes it.
            projected = comm.reduce_inputs_from_model_parallel_workers(projected)
            _accumulate(
                result,
                projected,
                ids,
                block.counts,
                weights=selected,
                fused=fused and unique,
            )
        return result

    @staticmethod
    @once_differentiable
    def backward(ctx, gradient):
        hidden, weights, order, prefixes, counts, *parameters = ctx.saved_tensors
        blocks, activation, fused, unique, topk, adapters, stride = ctx.execution_metadata
        device_type, enabled, compute_dtype = ctx.autocast
        hidden_grad = torch.zeros_like(hidden) if ctx.needs_input_grad[0] else None
        weight_grad = torch.zeros_like(weights) if ctx.needs_input_grad[1] else None
        detached = [
            p.detach().requires_grad_(need)
            for p, need in zip(parameters, ctx.parameter_needs_grad, strict=True)
        ]
        parameter_grads = [None] * len(parameters)
        for block, rng in reversed(tuple(zip(blocks, ctx.rng_states, strict=True))):
            tokens, selected, assignments, ids = _pack(
                hidden, weights, order, prefixes, counts, block, topk, fused
            )
            tokens.requires_grad_(hidden_grad is not None)
            local = [stride * owner + offset for owner in block.owners for offset in range(stride)]
            targets = ([tokens] if hidden_grad is not None else []) + [
                detached[i] for i in local if ctx.parameter_needs_grad[i]
            ]
            if fused:
                from .triton_routing import gather_gradient

                incoming = gather_gradient(gradient, ids)
            else:
                incoming = gradient[ids]
            with (
                torch.enable_grad(),
                torch.autocast(device_type, enabled=enabled, dtype=compute_dtype),
                tensor_parallel_rng_rewound_to(rng),
            ):
                projected = _project(
                    tokens,
                    detached,
                    block,
                    activation,
                    False,
                    False,
                    compute_dtype,
                    adapters,
                    stride,
                )
                projected = comm.reduce_inputs_from_model_parallel_workers(projected)
            grad_projected, grad_weights = _mixture_backward(projected.detach(), selected, incoming)
            if weight_grad is not None:
                weight_grad.index_copy_(0, assignments, grad_weights)
            grads = (
                iter(torch.autograd.grad(projected, targets, grad_projected))
                if targets
                else iter(())
            )
            if hidden_grad is not None:
                _accumulate(
                    hidden_grad,
                    next(grads),
                    ids,
                    block.counts,
                    fused=fused and unique,
                    reverse=True,
                )
            for i in local:
                if ctx.parameter_needs_grad[i]:
                    value = next(grads)
                    if parameter_grads[i] is None:
                        parameter_grads[i] = value
                    else:
                        parameter_grads[i].add_(value)
            del projected, grad_projected, grad_weights, incoming, targets, tokens, selected, grads
        return hidden_grad, weight_grad, None, None, None, None, *parameter_grads


def model_dtype_experts(x, indices, weights, experts):
    """Keep a single model-dtype accumulator and recompute one bounded group."""
    config = experts[0].config.model.moe
    if experts[0].up_proj.bias is not None or experts[0].down_proj.bias is not None:
        raise ValueError("model expert accumulation currently requires bias-free projections")
    hidden = x.reshape(-1, x.size(-1))
    flat = indices.flatten()
    if not flat.numel():
        return x * 0
    fused = config.blockwise_backend == "triton" and hidden.is_cuda
    if fused and torch.are_deterministic_algorithms_enabled():
        raise RuntimeError("Triton routing requires scheduled for deterministic execution")
    sizes = torch.bincount(flat, minlength=len(experts))
    sorted_ids = indices.sort(dim=-1).values
    unique = (sorted_ids[..., 1:] != sorted_ids[..., :-1]).all()
    statistics = torch.cat((sizes, unique.long().reshape(1))).tolist()
    sizes, unique = statistics[:-1], bool(statistics[-1])
    active = [i for i, count in enumerate(sizes) if count]
    active_sizes = [sizes[i] for i in active]
    counts = torch.tensor(active_sizes, device=x.device, dtype=torch.long)
    prefixes = counts.cumsum(0) - counts
    order = flat.argsort(stable=True)
    groups = plan_execution_groups(active_sizes, config.grouped_token_budget)
    blocks = tuple(_Block(g.experts, g.start, g.counts) for g in groups)
    parameters, adapters, stride = expert_parameters(experts, active)
    result = _ModelDtypeExperts.apply(
        comm.copy_inputs_to_model_parallel_workers(hidden),
        # Expert outputs are TP-reduced inside each group. Mixture gradients
        # are already complete replicas and must not be reduced a second time.
        weights.flatten(),
        order,
        prefixes,
        counts,
        (blocks, experts[0].activation, fused, unique, indices.size(-1), adapters, stride),
        *parameters,
    )
    return result.view_as(x)
