# Copyright (c) 2025-2026 Jaegeun Han
#
# SPDX-License-Identifier: Apache-2.0
"""
Shared AdamW update logic for tiled/host-offloaded optimizer states.

Used by both AdamWOptimizer.step() and MuonOptimizer._step_adamw().
This avoids DRY violations between the two codepaths.

Two compute paths:
  - CPU-compute path (params on GPU, M1-only): runs AdamW math on CPU,
    transfers only grad (GPU->CPU) and delta (CPU->GPU). Minimizes VRAM.
  - GPU-compute path (params on CPU, M2 active): states stay on CPU,
    .to() is a no-op, math runs on CPU anyway. Used when weight streaming
    has moved params to CPU.

When params are on GPU (M1 without M2), the CPU-compute path saves ~4x
transient VRAM per parameter compared to staging optimizer states on GPU.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from collections.abc import MutableMapping


def _should_offload_param(p: torch.nn.Parameter, min_param_elements: int) -> bool:
    """Check if a parameter should have its optimizer states offloaded to host."""
    if not getattr(p, "offloadable", True):
        return False
    return p.numel() >= min_param_elements


def _create_offloaded_state(
    p: torch.nn.Parameter,
    state: MutableMapping,
    state_dtype: torch.dtype,
    amsgrad: bool,
) -> None:
    """Initialize optimizer states on CPU for offloaded parameters."""
    state["step"] = 0
    state["exp_avg"] = torch.zeros_like(p, dtype=state_dtype, device="cpu")
    state["exp_avg_sq"] = torch.zeros_like(p, dtype=state_dtype, device="cpu")
    if amsgrad:
        state["max_exp_avg_sq"] = torch.zeros_like(p, dtype=state_dtype, device="cpu")


def _adamw_offloaded_step_cpu_compute(
    p: torch.nn.Parameter,
    grad: torch.Tensor,
    state: MutableMapping,
    lr: float,
    beta1: float,
    beta2: float,
    eps: float,
    weight_decay: float,
    amsgrad: bool,
    state_dtype: torch.dtype,
) -> None:
    """
    Run AdamW update on CPU when params are on GPU (M1-only, no weight streaming).

    Transfers grad GPU->CPU, runs AdamW math on CPU, computes delta, transfers
    delta CPU->GPU, applies delta to param.data. States never leave CPU.

    VRAM cost: one param-sized buffer for the delta (in p.dtype), instead of
    2x param-sized buffers for exp_avg + exp_avg_sq (in fp32).
    """
    if len(state) == 0:
        _create_offloaded_state(p, state, state_dtype, amsgrad)

    state["step"] += 1

    # Transfer gradient GPU -> CPU (single D2H transfer)
    grad_cpu_f32 = grad.to(device="cpu", dtype=torch.float32, non_blocking=False)

    # States are already on CPU — upgrade to float32 for accumulation
    exp_avg = state["exp_avg"].to(dtype=torch.float32)
    exp_avg_sq = state["exp_avg_sq"].to(dtype=torch.float32)
    max_exp_avg_sq = None
    if amsgrad:
        max_exp_avg_sq = state["max_exp_avg_sq"].to(dtype=torch.float32)

    # AdamW math on CPU (PyTorch ops use SIMD/AVX-512 via MKL/OpenBLAS)
    exp_avg.mul_(beta1).add_(grad_cpu_f32, alpha=1 - beta1)
    exp_avg_sq.mul_(beta2).addcmul_(grad_cpu_f32, grad_cpu_f32, value=1 - beta2)

    if amsgrad and max_exp_avg_sq is not None:
        torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
        denom = max_exp_avg_sq.sqrt().add_(eps)
    else:
        denom = exp_avg_sq.sqrt().add_(eps)

    bias_correction1 = 1.0 - beta1 ** state["step"]
    bias_correction2 = 1.0 - beta2 ** state["step"]
    step_size = lr * math.sqrt(bias_correction2) / bias_correction1

    # Compute update delta on CPU, cast to param dtype for smaller H2D transfer
    delta = (exp_avg / denom).mul_(-step_size).to(p.dtype)

    # Weight decay applied directly on GPU (scalar multiply, no transfer needed)
    if weight_decay != 0:
        p.data.mul_(1 - lr * weight_decay)

    # Transfer delta CPU -> GPU (single H2D, in param dtype)
    p.data.add_(delta.to(device=p.data.device, non_blocking=False))

    # Write updated states back in storage dtype (they never left CPU)
    state["exp_avg"] = exp_avg.to(dtype=state_dtype)
    state["exp_avg_sq"] = exp_avg_sq.to(dtype=state_dtype)
    if amsgrad and max_exp_avg_sq is not None:
        state["max_exp_avg_sq"] = max_exp_avg_sq.to(dtype=state_dtype)


def _adamw_offloaded_step(
    p: torch.nn.Parameter,
    grad: torch.Tensor,
    state: MutableMapping,
    lr: float,
    beta1: float,
    beta2: float,
    eps: float,
    weight_decay: float,
    amsgrad: bool,
    state_dtype: torch.dtype,
) -> None:
    """
    Run AdamW update with optimizer states on host (CPU).

    When params are on GPU (M1-only, no weight streaming): delegates to
    _adamw_offloaded_step_cpu_compute which runs math on CPU, avoiding
    the VRAM spike from staging optimizer states on GPU.

    When params are on CPU (weight streaming active): runs math on CPU
    natively since states and params are already there.
    """
    if len(state) == 0:
        _create_offloaded_state(p, state, state_dtype, amsgrad)

    state["step"] += 1

    compute_device = p.data.device

    # When params are on GPU (M1-only), run AdamW on CPU to avoid
    # staging optimizer states on GPU (which would spike peak VRAM).
    if compute_device.type == "cuda":
        _adamw_offloaded_step_cpu_compute(
            p, grad, state, lr, beta1, beta2, eps, weight_decay, amsgrad, state_dtype
        )
        return

    # Params on CPU (weight streaming active): states stay on CPU,
    # .to() is a no-op, math runs on CPU.
    # Cast to float32 for accumulation stability
    exp_avg = state["exp_avg"].to(device=compute_device, dtype=torch.float32, non_blocking=False)
    exp_avg_sq = state["exp_avg_sq"].to(
        device=compute_device, dtype=torch.float32, non_blocking=False
    )
    max_exp_avg_sq = None
    if amsgrad:
        max_exp_avg_sq = state["max_exp_avg_sq"].to(
            device=compute_device, dtype=torch.float32, non_blocking=False
        )

    try:
        # Standard AdamW math (same as in-VRAM path) in float32
        # Use a float32 copy of gradient for accumulation if needed
        grad_f32 = grad.to(torch.float32)

        exp_avg.mul_(beta1).add_(grad_f32, alpha=1 - beta1)
        exp_avg_sq.mul_(beta2).addcmul_(grad_f32, grad_f32, value=1 - beta2)

        if amsgrad and max_exp_avg_sq is not None:
            torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
            denom = max_exp_avg_sq.sqrt().add_(eps)
        else:
            denom = exp_avg_sq.sqrt().add_(eps)

        bias_correction1 = 1.0 - beta1 ** state["step"]
        bias_correction2 = 1.0 - beta2 ** state["step"]
        step_size = lr * math.sqrt(bias_correction2) / bias_correction1

        if weight_decay != 0:
            p.data.mul_(1 - lr * weight_decay)

        # Cast exp_avg/denom back to p.dtype for final update if p is not float32
        p.data.addcdiv_(exp_avg.to(p.dtype), denom.to(p.dtype), value=-step_size)

    finally:
        # D2H: write states back to host in storage dtype
        state["exp_avg"] = exp_avg.to(device="cpu", dtype=state_dtype, non_blocking=False)
        state["exp_avg_sq"] = exp_avg_sq.to(device="cpu", dtype=state_dtype, non_blocking=False)
        if amsgrad and max_exp_avg_sq is not None:
            state["max_exp_avg_sq"] = max_exp_avg_sq.to(
                device="cpu", dtype=state_dtype, non_blocking=False
            )
