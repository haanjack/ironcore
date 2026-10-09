# Copyright (c) 2025-2026 Jaegeun Han
# SPDX-License-Identifier: Apache-2.0

"""Bounded expert recomputation with one layer-wide input-gradient accumulator."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.autograd.function import once_differentiable

from ironcore.parallel.tensor_parallel import comm

from .execution_groups import plan_execution_groups


@dataclass(frozen=True)
class _Block:
    owners: tuple[int, ...]
    start: int
    counts: tuple[int, ...]
    width: int = 0
    unique_tokens: bool = False


def _pack(
    hidden: torch.Tensor,
    weights: torch.Tensor,
    order: torch.Tensor,
    prefixes: torch.Tensor,
    counts: torch.Tensor,
    block: _Block,
    topk: int,
    fused: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if fused:
        from .triton_routing import pack

        return pack(
            hidden,
            weights,
            order,
            prefixes,
            counts,
            block.start,
            block.width,
            topk,
            rows=sum(block.counts),
        )
    if block.width:
        lanes = torch.arange(block.width, device=hidden.device)
        positions = prefixes[:, None] + block.start + lanes
        valid = block.start + lanes < counts[:, None]
        positions = positions.masked_fill(~valid, 0).flatten()
        assignments = order[positions].masked_fill(~valid.flatten(), -1)
    else:
        positions = torch.arange(sum(block.counts), device=hidden.device) + block.start
        assignments = order[positions]
    valid = assignments >= 0
    safe = assignments.clamp_min(0)
    token_ids = (safe // topk).masked_fill(~valid, -1)
    tokens = hidden[token_ids.clamp_min(0)].masked_fill(~valid[:, None], 0)
    selected = weights[safe].masked_fill(~valid, 0)
    return tokens, selected, assignments, token_ids


def _scatter(
    destination: torch.Tensor,
    values: torch.Tensor,
    token_ids: torch.Tensor,
    fused: bool,
    unique: bool = False,
) -> None:
    if fused:
        from .triton_routing import scatter

        scatter(destination, values, token_ids, unique=unique)
    else:
        destination.index_add_(
            0,
            token_ids.clamp_min(0),
            values.to(destination.dtype).masked_fill(token_ids[:, None] < 0, 0),
        )


def _project(tokens, parameters, block, activation, bias, padded, compute_dtype):
    stride = 3 if bias else 2
    up = torch.stack([parameters[stride * i] for i in block.owners])
    down = torch.stack([parameters[stride * i + 1] for i in block.owners])
    if padded:
        packed = tokens.view(len(block.owners), block.width, tokens.size(-1))
        projected = torch.bmm(packed, up)
        if bias:
            projected = (
                projected
                + torch.stack([parameters[stride * i + 2] for i in block.owners])[:, None, :]
            )
        return torch.bmm(activation(projected), down).reshape_as(tokens).to(tokens.dtype)
    from .grouped import _grouped_mm

    offsets = torch.tensor(block.counts, device=tokens.device, dtype=torch.int32).cumsum(
        0, dtype=torch.int32
    )
    projected = _grouped_mm(tokens.to(compute_dtype), up.to(compute_dtype), offsets, block.counts)
    if bias:
        selected = torch.repeat_interleave(
            torch.arange(len(block.owners), device=tokens.device),
            torch.tensor(block.counts, device=tokens.device),
            output_size=tokens.size(0),
        )
        projected = (
            projected + torch.stack([parameters[stride * i + 2] for i in block.owners])[selected]
        )
    return _grouped_mm(
        activation(projected).to(compute_dtype), down.to(compute_dtype), offsets, block.counts
    )


class _ScheduledExperts(torch.autograd.Function):
    """Return parameter VJPs once, without nested graphs touching original leaves.

    Block recomputation differentiates detached small input tiles and detached
    Parameter views. Only this outer Function returns gradients to the original
    Parameters, preserving DDP hooks and unused-expert behavior. First-order only.
    """

    @staticmethod
    def forward(ctx, hidden, weights, order, prefixes, counts, metadata, *parameters):
        blocks, activation, bias, padded, topk, fused = metadata
        device_type = hidden.device.type
        enabled = torch.is_autocast_enabled(device_type)
        compute_dtype = torch.get_autocast_dtype(device_type) if enabled else hidden.dtype
        value_dtype = hidden.dtype if padded else compute_dtype
        output_dtype = torch.promote_types(value_dtype, weights.dtype)
        if hidden.is_cuda and enabled and output_dtype in {torch.float16, torch.bfloat16}:
            output_dtype = torch.float32
        accumulator_dtype = (
            torch.float32 if output_dtype in {torch.float16, torch.bfloat16} else output_dtype
        )
        result = torch.zeros_like(hidden, dtype=accumulator_dtype)
        ctx.save_for_backward(hidden, weights, order, prefixes, counts, *parameters)
        ctx.execution_metadata = metadata
        ctx.autocast = (device_type, enabled, compute_dtype)
        ctx.parameter_needs_grad = tuple(p.requires_grad for p in parameters)
        for block in blocks:
            tokens, selected, assignments, token_ids = _pack(
                hidden, weights, order, prefixes, counts, block, topk, fused
            )
            projected = _project(tokens, parameters, block, activation, bias, padded, compute_dtype)
            if fused:
                from .triton_routing import weighted_scatter

                weighted_scatter(result, projected, selected, token_ids, unique=block.unique_tokens)
            else:
                _scatter(result, projected * selected[:, None], token_ids, False)
        return result.to(output_dtype)

    @staticmethod
    @once_differentiable
    def backward(ctx, gradient):
        hidden, weights, order, prefixes, counts, *parameters = ctx.saved_tensors
        blocks, activation, bias, padded, topk, fused = ctx.execution_metadata
        device_type, enabled, compute_dtype = ctx.autocast
        hidden_grad = (
            torch.zeros_like(
                hidden,
                dtype=torch.float32
                if hidden.dtype in {torch.float16, torch.bfloat16}
                else hidden.dtype,
            )
            if ctx.needs_input_grad[0]
            else None
        )
        weight_grad = torch.zeros_like(weights) if ctx.needs_input_grad[1] else None
        detached = [
            p.detach().requires_grad_(need)
            for p, need in zip(parameters, ctx.parameter_needs_grad, strict=True)
        ]
        parameter_grads = [None] * len(parameters)
        stride = 3 if bias else 2
        # Match checkpoint backward's reverse block accumulation order.
        for block in reversed(blocks):
            tokens, selected, assignments, token_ids = _pack(
                hidden, weights, order, prefixes, counts, block, topk, fused
            )
            tokens.requires_grad_(hidden_grad is not None)
            selected.requires_grad_(weight_grad is not None)
            local = [stride * owner + offset for owner in block.owners for offset in range(stride)]
            targets = []
            if hidden_grad is not None:
                targets.append(tokens)
            if weight_grad is not None:
                targets.append(selected)
            targets.extend(detached[i] for i in local if ctx.parameter_needs_grad[i])
            if fused:
                from .triton_routing import gather_gradient

                incoming = gather_gradient(gradient, token_ids)
            else:
                incoming = gradient[token_ids.clamp_min(0)].masked_fill(token_ids[:, None] < 0, 0)
            with (
                torch.enable_grad(),
                torch.autocast(device_type, enabled=enabled, dtype=compute_dtype),
            ):
                projected = _project(
                    tokens, detached, block, activation, bias, padded, compute_dtype
                )
                weighted = projected * selected[:, None]
                grads = iter(torch.autograd.grad(weighted, targets, incoming.to(weighted.dtype)))
            if hidden_grad is not None:
                _scatter(hidden_grad, next(grads), token_ids, fused, block.unique_tokens)
            if weight_grad is not None:
                values = next(grads)
                if fused:
                    from .triton_routing import store_weight_gradient

                    store_weight_gradient(weight_grad, values, assignments)
                else:
                    valid = assignments >= 0
                    weight_grad.index_copy_(0, assignments[valid], values[valid])
            for i in local:
                if ctx.parameter_needs_grad[i]:
                    value = next(grads)
                    if parameter_grads[i] is None:
                        parameter_grads[i] = value
                    else:
                        parameter_grads[i].add_(value)
            # Do not overlap two blocks' GEMM intermediates or temporary gradients.
            del projected, weighted, incoming, targets, tokens, selected, grads
        if hidden_grad is not None:
            hidden_grad = hidden_grad.to(hidden.dtype)
        return hidden_grad, weight_grad, None, None, None, None, *parameter_grads


def scheduled_experts(
    x: torch.Tensor,
    indices: torch.Tensor,
    weights: torch.Tensor,
    experts: torch.nn.ModuleList,
    *,
    padded: bool,
) -> torch.Tensor:
    """Recompute batched or jagged GEMMs with bounded local input derivatives."""
    hidden = x.reshape(-1, x.size(-1))
    flat = indices.flatten()
    if not flat.numel():
        return x * 0
    config = experts[0].config
    activation = experts[0].activation
    if next(activation.parameters(), None) is not None:
        raise ValueError("Scheduled experts require parameter-free activations")
    fused = config.model.moe.blockwise_backend == "triton" and hidden.is_cuda
    if fused and torch.are_deterministic_algorithms_enabled():
        raise RuntimeError(
            "Triton routing uses atomic scatter; choose scheduled for deterministic execution"
        )
    if fused and hidden.dtype not in {torch.float16, torch.bfloat16, torch.float32}:
        raise ValueError("Triton routing supports FP32, FP16 and BF16 inputs")
    counts_tensor = torch.bincount(flat, minlength=len(experts))
    unique_topk = False
    if fused and not padded:
        sorted_ids = indices.sort(dim=-1).values
        unique = (sorted_ids[..., 1:] != sorted_ids[..., :-1]).all()
        # Piggyback the uniqueness check on the existing expert-count CPU transfer.
        statistics = torch.cat((counts_tensor, unique.long().reshape(1))).tolist()
        sizes, unique_topk = statistics[:-1], bool(statistics[-1])
    else:
        sizes = counts_tensor.tolist()
    active = [i for i, count in enumerate(sizes) if count]
    active_sizes = [sizes[i] for i in active]
    order = flat.argsort(stable=True)
    counts = torch.tensor(active_sizes, device=x.device, dtype=torch.long)
    prefixes = counts.cumsum(0) - counts
    if padded:
        width = min(max(active_sizes), config.trainer.mlp_chunk_size)
        # Keep tail tiles at the same bounded width, avoiding new pack-kernel
        # variants as the most-loaded expert's last rows change between updates.
        blocks = tuple(
            _Block(
                tuple(range(len(active))),
                start,
                tuple(active_sizes),
                width,
            )
            for start in range(0, max(active_sizes), width)
        )
    else:
        groups = plan_execution_groups(active_sizes, config.model.moe.grouped_token_budget)
        blocks = tuple(
            _Block(g.experts, g.start, g.counts, unique_tokens=unique_topk and len(g.experts) == 1)
            for g in groups
        )
    bias = experts[0].up_proj.bias is not None
    parameters = tuple(
        parameter
        for i in active
        for parameter in (
            (experts[i].up_proj.weight, experts[i].down_proj.weight, experts[i].up_proj.bias)
            if bias
            else (experts[i].up_proj.weight, experts[i].down_proj.weight)
        )
    )
    parallel_hidden = comm.copy_inputs_to_model_parallel_workers(hidden)
    parallel_weights = comm.copy_inputs_to_model_parallel_workers(weights).flatten()
    result = _ScheduledExperts.apply(
        parallel_hidden,
        parallel_weights,
        order,
        prefixes,
        counts,
        (blocks, activation, bias, padded, indices.size(-1), fused),
        *parameters,
    )
    result = comm.reduce_inputs_from_model_parallel_workers(result)
    if experts[0].down_proj.bias is not None:
        biases = torch.stack(
            [
                e.down_proj.bias if sizes[i] else e.down_proj.bias.detach()
                for i, e in enumerate(experts)
            ]
        )
        result = result + (
            biases[indices.reshape(-1, indices.size(-1))] * weights.reshape(-1, indices.size(-1), 1)
        ).sum(1)
    return result.view_as(x)
