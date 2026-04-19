# Copyright (c) 2025-2026 Jaegeun Han
#
# SPDX-License-Identifier: Apache-2.0
"""
Shared AdamW update logic for tiled/host-offloaded optimizer states.

Used by both AdamWOptimizer.step() and MuonOptimizer._step_adamw().
This avoids DRY violations between the two codepaths.

M1 approach: optimizer states (exp_avg, exp_avg_sq) live in pageable host
memory. Before each update, they transfer to GPU synchronously. After the
update, they transfer back to CPU. No pinned memory, no tiling, no async DMA.
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

    Flow:
      1. exp_avg, exp_avg_sq start on CPU
      2. Transfer to GPU (sync H2D)
      3. Run standard AdamW math on GPU
      4. Transfer updated states back to CPU (sync D2H)
      5. Update p.data in-place on GPU
    """
    if len(state) == 0:
        _create_offloaded_state(p, state, state_dtype, amsgrad)

    state["step"] += 1

    # H2D: bring states to GPU for the update
    gpu_device = p.data.device
    exp_avg = state["exp_avg"].to(gpu_device, non_blocking=False)
    exp_avg_sq = state["exp_avg_sq"].to(gpu_device, non_blocking=False)
    max_exp_avg_sq = None
    if amsgrad:
        max_exp_avg_sq = state["max_exp_avg_sq"].to(gpu_device, non_blocking=False)

    try:
        # Standard AdamW math (same as in-VRAM path)
        exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
        exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

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

        p.data.addcdiv_(exp_avg, denom, value=-step_size)

    finally:
        # D2H: always write states back to host, even on error.
        # Prevents GPU memory leaks when M2+ pinned pool is in use.
        state["exp_avg"] = exp_avg.to("cpu", non_blocking=False)
        state["exp_avg_sq"] = exp_avg_sq.to("cpu", non_blocking=False)
        if amsgrad and max_exp_avg_sq is not None:
            state["max_exp_avg_sq"] = max_exp_avg_sq.to("cpu", non_blocking=False)
