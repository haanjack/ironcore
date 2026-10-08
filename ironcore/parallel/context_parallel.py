# Copyright (c) 2025-2026 Jaegeun Han
# SPDX-License-Identifier: Apache-2.0

"""Sequence ownership, differentiable token means and CP replica gradients."""

from __future__ import annotations

import torch
import torch.distributed as dist
import torch.nn.functional as F

from . import parallel_states as ps


class _GatherSequence(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value: torch.Tensor, reduce_backward: bool) -> torch.Tensor:
        ctx.group = ps.get_context_parallel_group()
        ctx.rank = ps.get_context_parallel_rank()
        ctx.size = ps.get_context_parallel_world_size()
        ctx.reduce_backward = reduce_backward
        pieces = [torch.empty_like(value) for _ in range(ctx.size)]
        dist.all_gather(pieces, value.contiguous(), group=ctx.group)
        return torch.cat(pieces, dim=1)

    @staticmethod
    def backward(ctx, gradient: torch.Tensor) -> tuple[torch.Tensor, None]:
        if ctx.reduce_backward:
            # Reference K/V gathering: every query shard contributes to each
            # owner's K/V gradient. Output reconstruction only splits gradients.
            gradient = gradient.contiguous().clone()
            dist.all_reduce(gradient, group=ctx.group)
        return gradient.chunk(ctx.size, dim=1)[ctx.rank].contiguous(), None


def gather_context_parallel(value: torch.Tensor, *, reduce_backward: bool = False) -> torch.Tensor:
    """Reconstruct sequence order; optionally sum gradients from remote queries."""
    if ps.get_context_parallel_world_size() == 1:
        return value
    return _GatherSequence.apply(value, reduce_backward)


def partition_context_inputs(
    input_ids: torch.Tensor,
    labels: torch.Tensor | None,
    position_ids: torch.Tensor,
    pad_token_id: int = 0,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor, int]:
    """Right-pad then take a contiguous sequence shard; labels are already shifted."""
    size = ps.get_context_parallel_world_size()
    rank = ps.get_context_parallel_rank()
    length = input_ids.size(1)
    if length == 0:
        raise ValueError("Context parallel requires a nonempty sequence")
    if position_ids.shape != input_ids.shape or (
        labels is not None and labels.shape != input_ids.shape
    ):
        raise ValueError("Context parallel input, label and position shapes must match")
    padding = (-length) % size
    if padding:
        input_ids = F.pad(input_ids, (0, padding), value=pad_token_id)
        if labels is not None:
            labels = F.pad(labels, (0, padding), value=-100)
        # Padding cannot influence real causal queries and has no loss. Reuse
        # the final position so a maximum-length sequence never indexes past
        # the positional embedding cache after CP divisibility padding.
        suffix = position_ids[:, -1:].expand(-1, padding)
        position_ids = torch.cat((position_ids, suffix), dim=1)
    start = rank * (input_ids.size(1) // size)
    end = start + input_ids.size(1) // size
    return (
        input_ids[:, start:end].contiguous(),
        labels[:, start:end].contiguous() if labels is not None else None,
        position_ids[:, start:end].contiguous(),
        length,
    )


class _TokenMean(torch.autograd.Function):
    @staticmethod
    def forward(ctx, numerator: torch.Tensor, count: torch.Tensor) -> torch.Tensor:
        totals = torch.stack((numerator.detach().double(), count.detach().double()))
        dist.all_reduce(totals, group=ps.get_context_parallel_group())
        # A complete DP batch may be empty while other DP peers have valid
        # tokens. It must still run backward/DDP collectives with zero gradients.
        # The trainer rejects an update only when its global DP count is zero.
        denominator = totals[1].clamp_min(1)
        ctx.denominator = denominator.to(numerator.dtype)
        return (totals[0] / denominator).to(numerator.dtype)

    @staticmethod
    def backward(ctx, gradient: torch.Tensor) -> tuple[torch.Tensor, None]:
        # Each replica contributes its local numerator; parameter gradients are
        # summed over CP before clipping. Do not average this gradient twice.
        return gradient / ctx.denominator, None


def context_parallel_token_mean(losses: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Report the same global token mean on every CP rank with local gradients."""
    numerator = (losses.float() * mask.float()).sum()
    count = mask.float().sum()
    if ps.get_context_parallel_world_size() == 1:
        return numerator / count.clamp_min(1)
    return _TokenMean.apply(numerator, count)


class _SumWithLocalGradient(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value: torch.Tensor) -> torch.Tensor:
        result = value.clone()
        dist.all_reduce(result, group=ps.get_context_parallel_group())
        return result

    @staticmethod
    def backward(ctx, gradient: torch.Tensor) -> torch.Tensor:
        # CP replica parameter gradients are summed once by the trainer.
        return gradient


def sum_context_parallel_statistics(value: torch.Tensor) -> torch.Tensor:
    """Sum small routing statistics while preserving only local derivatives."""
    if ps.get_context_parallel_world_size() == 1:
        return value
    return _SumWithLocalGradient.apply(value)


def synchronize_context_parallel_gradients(model: torch.nn.Module) -> None:
    """Sum token-shard contributions after DP synchronization and AMP unscale."""
    if ps.get_context_parallel_world_size() == 1:
        return
    group = ps.get_context_parallel_group()
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not parameters:
        return
    present = torch.tensor(
        [parameter.grad is not None for parameter in parameters],
        dtype=torch.int32,
        device=parameters[0].device,
    )
    dist.all_reduce(present, op=dist.ReduceOp.MAX, group=group)
    for parameter, used in zip(parameters, present.tolist(), strict=True):
        if not used:
            # Optimizers skip globally unused parameters. Giving them a zero
            # gradient would incorrectly create moments and apply weight decay.
            continue
        if parameter.grad is None:
            parameter.grad = torch.zeros_like(parameter)
        dist.all_reduce(parameter.grad, group=group)
