# Copyright (c) 2025-2026 Jaegeun Han
# SPDX-License-Identifier: Apache-2.0
"""CUDA RMSNorm: FP32 register statistics, direct low-precision output storage.

Used for frozen scale weights (LoRA/inference). Trainable scales and FP32/CPU
inputs keep the ordinary autograd reference, including its higher derivatives.
"""

import torch
from torch.autograd.function import once_differentiable

try:
    import triton
    import triton.language as tl
except ImportError:
    triton = None


if triton is not None:

    @triton.jit
    def _torch_row_sum(
        x,
        gradient,
        weight,
        row,
        channels: tl.constexpr,
        lanes: tl.constexpr,
        squared: tl.constexpr,
        with_scale: tl.constexpr,
        round_before_scale: tl.constexpr,
    ):
        # ATen's contiguous FP32 row reduction loads vectors of four, keeps
        # four accumulators, combines them left-to-right, then reduces lanes.
        # Reproduce that ordering without materializing x.float()/x.square().
        lane = tl.arange(0, lanes)
        component = tl.arange(0, 4)
        accumulated = tl.full((lanes, 4), 0, tl.float32)
        for step in range(tl.cdiv(channels, 4 * lanes)):
            col = lane[:, None] * 4 + component[None, :] + step * 4 * lanes
            value = tl.load(x + row * channels + col, col < channels, other=0).to(tl.float32)
            if squared:
                term = value * value
            else:
                grad = tl.load(gradient + row * channels + col, col < channels, other=0).to(
                    tl.float32
                )
                if with_scale:
                    scale = tl.load(weight + col, col < channels, other=0).to(tl.float32)
                    grad = grad * scale
                    if round_before_scale:
                        grad = grad.to(x.dtype.element_ty).to(tl.float32)
                term = value * grad
            accumulated = accumulated + term
        even, odd = tl.split(accumulated.reshape(lanes, 2, 2))
        a, c = tl.split(even)
        b, d = tl.split(odd)
        combined = ((a + b) + c) + d
        # ATen combines block.x groups of 32 before its intra-warp reduction.
        if lanes > 32:
            combined = tl.sum(combined.reshape(lanes // 32, 32), axis=0)
        return tl.sum(combined, axis=0)

    @triton.jit
    def _forward(
        x,
        weight,
        output,
        inv_rms,
        channels: tl.constexpr,
        epsilon: tl.constexpr,
        tile: tl.constexpr,
        with_scale: tl.constexpr,
        round_before_scale: tl.constexpr,
        torch_compatible: tl.constexpr,
        lanes: tl.constexpr,
    ):
        row = tl.program_id(0)
        col = tl.arange(0, tile)
        value = tl.load(x + row * channels + col, mask=col < channels, other=0).to(tl.float32)
        if torch_compatible:
            total = _torch_row_sum(x, x, weight, row, channels, lanes, True, False, False)
        else:
            total = tl.sum(value * value, axis=0)
        inverse = tl.rsqrt(total / channels + epsilon)
        normed = value * inverse
        if round_before_scale:
            normed = normed.to(x.dtype.element_ty).to(tl.float32)
        if with_scale:
            scale = tl.load(weight + col, mask=col < channels, other=0).to(tl.float32)
            normed = normed * scale
        tl.store(
            output + row * channels + col, normed.to(output.dtype.element_ty), mask=col < channels
        )
        tl.store(inv_rms + row, inverse)

    @triton.jit
    def _backward(
        x,
        weight,
        inv_rms,
        gradient,
        dx,
        channels: tl.constexpr,
        tile: tl.constexpr,
        with_scale: tl.constexpr,
        round_before_scale: tl.constexpr,
        torch_compatible: tl.constexpr,
        lanes: tl.constexpr,
    ):
        row = tl.program_id(0)
        col = tl.arange(0, tile)
        value = tl.load(x + row * channels + col, mask=col < channels, other=0).to(tl.float32)
        inverse = tl.load(inv_rms + row)
        grad = tl.load(gradient + row * channels + col, mask=col < channels, other=0).to(tl.float32)
        if with_scale:
            scale = tl.load(weight + col, mask=col < channels, other=0).to(tl.float32)
            grad = grad * scale
            if round_before_scale:
                grad = grad.to(x.dtype.element_ty).to(tl.float32)
        if torch_compatible:
            d_inverse = _torch_row_sum(
                x, gradient, weight, row, channels, lanes, False, with_scale, round_before_scale
            )
            # Preserve the rsqrt -> mean -> square autograd rounding boundaries.
            d_variance = (d_inverse * -0.5) * ((inverse * inverse) * inverse)
            result = grad * inverse + (value * 2.0) * (d_variance / channels)
        else:
            correction = tl.sum(value * grad, axis=0) / channels
            result = inverse * grad - value * (inverse * inverse * inverse) * correction
        tl.store(dx + row * channels + col, result.to(dx.dtype.element_ty), mask=col < channels)


class _FrozenScaleRMSNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, epsilon, round_before_scale, torch_compatible):
        contiguous = x.contiguous()
        output = torch.empty_like(contiguous)
        channels = x.size(-1)
        inverse = torch.empty(x.numel() // channels, dtype=torch.float32, device=x.device)
        # Match ATen's 512-thread FP32 reduce launch geometry. Other widths
        # retain the existing fused path rather than claiming exact ordering.
        compatible = torch_compatible and 128 <= channels < 8192 and channels % 4 == 0
        dim0 = 1 << (max(1, channels // 4).bit_length() - 1)
        width = min(dim0, 32)
        height = min(1 << (inverse.numel().bit_length() - 1), 512 // width)
        lanes = min(dim0, 512 // height)
        _forward[(inverse.numel(),)](
            contiguous,
            weight if weight is not None else contiguous,
            output,
            inverse,
            channels,
            epsilon,
            triton.next_power_of_2(channels),
            weight is not None,
            round_before_scale,
            compatible,
            lanes,
            enable_fp_fusion=False,
        )
        ctx.save_for_backward(contiguous, weight, inverse)
        ctx.round_before_scale = round_before_scale
        ctx.torch_compatible = compatible
        ctx.lanes = lanes
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, gradient):
        x, weight, inverse = ctx.saved_tensors
        dx = torch.empty_like(x)
        _backward[(inverse.numel(),)](
            x,
            weight if weight is not None else x,
            inverse,
            gradient.contiguous(),
            dx,
            x.size(-1),
            triton.next_power_of_2(x.size(-1)),
            weight is not None,
            ctx.round_before_scale,
            ctx.torch_compatible,
            ctx.lanes,
            enable_fp_fusion=False,
        )
        return dx, None, None, None, None


def frozen_scale_rms_norm(x, weight, epsilon, *, round_before_scale, torch_compatible=False):
    """Return None if this input needs the regular autograd implementation."""
    if (
        triton is None
        or not x.is_cuda
        or x.numel() == 0
        or x.dtype not in (torch.float16, torch.bfloat16)
        or x.size(-1) > 16384
        or (weight is not None and weight.requires_grad)
        or (round_before_scale and weight is not None and weight.dtype != x.dtype)
    ):
        return None
    return _FrozenScaleRMSNorm.apply(x, weight, epsilon, round_before_scale, torch_compatible)
