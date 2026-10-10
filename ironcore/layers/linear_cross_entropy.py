# Copyright (c) 2025-2026 Jaegeun Han
# SPDX-License-Identifier: Apache-2.0
"""Recomputed token-chunk linear + vocabulary CE without full B*S*V logits."""

import torch
import torch.nn.functional as F

from ironcore.parallel.tensor_parallel import vocab_parallel_cross_entropy


def vocab_linear(hidden, weight, padding_start, *, transposed=False, trim_padding=False):
    """Preserve padded TP storage/logits, but exclude dummy rows from GEMM.

    Padding zeros still change a backward GEMM's contraction geometry. Project
    only real vocabulary entries, then restore dummy logits for the uniform
    TP gather/CE contract. Slicing preserves full weight-gradient shapes.
    """
    if not trim_padding or padding_start is None:
        return hidden @ weight if transposed else F.linear(hidden, weight)
    from ironcore.parallel import parallel_states

    vocab = weight.size(1 if transposed else 0)
    rank = parallel_states.get_tensor_model_parallel_rank()
    active = max(0, min(vocab, padding_start - rank * vocab))
    selected = weight[:, :active].T if transposed else weight[:active]
    output = F.linear(hidden, selected)
    return F.pad(output, (0, vocab - active), value=float("-inf")) if active < vocab else output


class _LinearCrossEntropy(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        hidden,
        weight,
        labels,
        chunk,
        padding_start,
        transposed,
        softcap,
        logits_scaling,
        trim_padding,
    ):
        ctx.save_for_backward(hidden, weight, labels)
        ctx.chunk, ctx.padding_start, ctx.transposed = chunk, padding_start, transposed
        ctx.softcap = softcap
        ctx.logits_scaling = logits_scaling
        ctx.trim_padding = trim_padding
        ctx.autocast = torch.is_autocast_enabled(hidden.device.type)
        ctx.autocast_dtype = torch.get_autocast_dtype(hidden.device.type)
        h = hidden.reshape(-1, hidden.size(-1))
        y = labels.reshape(-1)
        output = torch.empty(y.shape, device=hidden.device, dtype=torch.float32)
        for start in range(0, y.numel(), chunk):
            logits = vocab_linear(
                h[start : start + chunk],
                weight,
                padding_start,
                transposed=transposed,
                trim_padding=trim_padding,
            )
            if logits_scaling != 1.0:
                logits = logits / logits_scaling
            output[start : start + chunk] = vocab_parallel_cross_entropy(
                ((logits / softcap).tanh() * softcap if softcap is not None else logits).float(),
                y[start : start + chunk],
                padding_start_idx=padding_start,
            )
        return output.view_as(labels)

    @staticmethod
    def backward(ctx, gradient):
        hidden, weight, labels = ctx.saved_tensors
        h = hidden.reshape(-1, hidden.size(-1))
        y = labels.reshape(-1)
        dh = torch.empty_like(h)
        dw = torch.zeros_like(weight) if ctx.needs_input_grad[1] else None
        for start in range(0, y.numel(), ctx.chunk):
            with (
                torch.enable_grad(),
                torch.autocast(hidden.device.type, dtype=ctx.autocast_dtype, enabled=ctx.autocast),
            ):
                hs = h[start : start + ctx.chunk].detach().requires_grad_(True)
                ws = weight.detach().requires_grad_(ctx.needs_input_grad[1])
                logits = vocab_linear(
                    hs,
                    ws,
                    ctx.padding_start,
                    transposed=ctx.transposed,
                    trim_padding=ctx.trim_padding,
                )
                if ctx.logits_scaling != 1.0:
                    logits = logits / ctx.logits_scaling
                if ctx.softcap is not None:
                    logits = (logits / ctx.softcap).tanh() * ctx.softcap
                loss = vocab_parallel_cross_entropy(
                    logits.float(),
                    y[start : start + ctx.chunk],
                    padding_start_idx=ctx.padding_start,
                )
                grads = torch.autograd.grad(
                    loss,
                    (hs, ws) if dw is not None else (hs,),
                    gradient.reshape(-1)[start : start + ctx.chunk],
                )
            dh[start : start + ctx.chunk] = grads[0]
            if dw is not None:
                dw.add_(grads[1])
        return dh.view_as(hidden), dw, None, None, None, None, None, None, None


def linear_cross_entropy(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    labels: torch.Tensor,
    chunk: int,
    padding_start: int | None = None,
    transposed: bool = False,
    softcap: float | None = None,
    logits_scaling: float = 1.0,
    trim_padding: bool = False,
) -> torch.Tensor:
    """Recompute bounded vocabulary logits and preserve optional logit softcapping."""
    if chunk <= 0:
        raise ValueError("Linear CE chunk size must be positive")
    return _LinearCrossEntropy.apply(
        hidden,
        weight,
        labels,
        chunk,
        padding_start,
        transposed,
        softcap,
        logits_scaling,
        trim_padding,
    )
