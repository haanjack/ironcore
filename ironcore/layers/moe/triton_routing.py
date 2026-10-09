# Copyright (c) 2025-2026 Jaegeun Han
# SPDX-License-Identifier: Apache-2.0

"""Optional CUDA routing kernels, imported only for the explicit Triton backend."""

from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl
except ImportError as error:
    raise RuntimeError(
        "The triton blockwise backend requires Triton in the GPU container"
    ) from error


@triton.jit(do_not_specialize=["start"])
def _pack_kernel(
    hidden,
    weights,
    order,
    prefixes,
    counts,
    tokens,
    selected,
    assignments,
    token_ids,
    hidden_s0: tl.constexpr,
    hidden_s1: tl.constexpr,
    start,
    width: tl.constexpr,
    topk: tl.constexpr,
    channels: tl.constexpr,
    tile: tl.constexpr,
    padded: tl.constexpr,
):
    row = tl.program_id(0)
    columns = tl.arange(0, tile)
    if padded:
        expert = row // width
        lane = row % width
        valid = start + lane < tl.load(counts + expert)
        position = tl.load(prefixes + expert) + start + lane
    else:
        valid = True
        position = start + row
    assignment = tl.load(order + position, mask=valid, other=0)
    token = assignment // topk
    values = tl.load(
        hidden + token * hidden_s0 + columns * hidden_s1, mask=valid & (columns < channels), other=0
    )
    weight = tl.load(weights + assignment, mask=valid, other=0)
    tl.store(tokens + row * channels + columns, values, mask=columns < channels)
    tl.store(selected + row, weight)
    tl.store(assignments + row, tl.where(valid, assignment, -1))
    tl.store(token_ids + row, tl.where(valid, token, -1))


@triton.jit
def _scatter_kernel(
    destination,
    values,
    ids,
    weights,
    dest_s0: tl.constexpr,
    dest_s1: tl.constexpr,
    value_s0: tl.constexpr,
    value_s1: tl.constexpr,
    channels: tl.constexpr,
    tile: tl.constexpr,
    weighted: tl.constexpr,
    product_dtype: tl.constexpr,
    unique: tl.constexpr,
):
    row = tl.program_id(0)
    columns = tl.arange(0, tile)
    token = tl.load(ids + row)
    valid = (token >= 0) & (columns < channels)
    value = tl.load(values + row * value_s0 + columns * value_s1, mask=valid, other=0).to(
        tl.float32
    )
    if weighted:
        weight = tl.load(weights + row).to(tl.float32)
        # Preserve the multiplication's output rounding before FP32 accumulation.
        value = (value * weight).to(product_dtype).to(tl.float32)
    address = destination + token * dest_s0 + columns * dest_s1
    if unique:
        previous = tl.load(address, mask=valid, other=0)
        tl.store(address, previous + value, mask=valid)
    else:
        tl.atomic_add(address, value, mask=valid, sem="relaxed")


@triton.jit
def _gather_gradient_kernel(
    gradient,
    ids,
    result,
    s0: tl.constexpr,
    s1: tl.constexpr,
    channels: tl.constexpr,
    tile: tl.constexpr,
):
    row = tl.program_id(0)
    columns = tl.arange(0, tile)
    token = tl.load(ids + row)
    value = tl.load(
        gradient + token * s0 + columns * s1, mask=(token >= 0) & (columns < channels), other=0
    )
    tl.store(result + row * channels + columns, value, mask=columns < channels)


@triton.jit(do_not_specialize=["rows"])
def _store_weight_gradient_kernel(destination, values, ids, rows, tile: tl.constexpr):
    row = tl.program_id(0) * tile + tl.arange(0, tile)
    assignment = tl.load(ids + row, mask=row < rows, other=-1)
    value = tl.load(values + row, mask=row < rows, other=0)
    tl.store(destination + assignment, value, mask=(row < rows) & (assignment >= 0))


def pack(
    hidden: torch.Tensor,
    weights: torch.Tensor,
    order: torch.Tensor,
    prefixes: torch.Tensor,
    counts: torch.Tensor,
    start: int,
    width: int,
    topk: int,
    *,
    rows: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fuse permutation lookup, input gather, zero padding and mixture-weight gather."""
    rows = prefixes.numel() * width if width else rows
    if rows is None:
        raise ValueError("Jagged routing requires an explicit row count")
    tokens = hidden.new_empty((rows, hidden.size(1)))
    selected = weights.new_empty(rows)
    assignments = order.new_empty(rows)
    token_ids = order.new_empty(rows)
    _pack_kernel[(rows,)](
        hidden,
        weights,
        order,
        prefixes,
        counts,
        tokens,
        selected,
        assignments,
        token_ids,
        hidden.stride(0),
        hidden.stride(1),
        start,
        width,
        topk,
        hidden.size(1),
        triton.next_power_of_2(hidden.size(1)),
        bool(width),
    )
    return tokens, selected, assignments, token_ids


def _launch_scatter(destination, values, token_ids, weights=None, unique=False):
    dtype = (
        torch.promote_types(values.dtype, weights.dtype) if weights is not None else values.dtype
    )
    product_dtype = {
        torch.float32: tl.float32,
        torch.float16: tl.float16,
        torch.bfloat16: tl.bfloat16,
    }[dtype]
    _scatter_kernel[(values.size(0),)](
        destination,
        values,
        token_ids,
        weights,
        destination.stride(0),
        destination.stride(1),
        values.stride(0),
        values.stride(1),
        values.size(1),
        triton.next_power_of_2(values.size(1)),
        weights is not None,
        product_dtype,
        unique,
        enable_fp_fusion=not unique,
    )


def scatter(
    destination: torch.Tensor,
    values: torch.Tensor,
    token_ids: torch.Tensor,
    *,
    unique: bool = False,
) -> None:
    """Accumulate one block's input derivatives into the shared FP32 buffer."""
    _launch_scatter(destination, values, token_ids, unique=unique)


def weighted_scatter(
    destination: torch.Tensor,
    values: torch.Tensor,
    weights: torch.Tensor,
    token_ids: torch.Tensor,
    *,
    unique: bool = False,
) -> None:
    """Fuse mixture multiplication and scatter without a weighted-output allocation."""
    _launch_scatter(destination, values, token_ids, weights, unique)


def gather_gradient(gradient: torch.Tensor, token_ids: torch.Tensor) -> torch.Tensor:
    """Gather possibly strided upstream gradients and clear padded rows."""
    result = gradient.new_empty((token_ids.numel(), gradient.size(1)))
    _gather_gradient_kernel[(token_ids.numel(),)](
        gradient,
        token_ids,
        result,
        gradient.stride(0),
        gradient.stride(1),
        gradient.size(1),
        triton.next_power_of_2(gradient.size(1)),
    )
    return result


def store_weight_gradient(
    destination: torch.Tensor, values: torch.Tensor, assignments: torch.Tensor
) -> None:
    """Store each routing assignment's unique mixture-weight derivative."""
    _store_weight_gradient_kernel[(triton.cdiv(assignments.numel(), 256),)](
        destination,
        values,
        assignments,
        assignments.numel(),
        256,
    )
