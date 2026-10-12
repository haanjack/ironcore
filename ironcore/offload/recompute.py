# Copyright (c) 2025-2026 Jaegeun Han
# SPDX-License-Identifier: Apache-2.0
"""CPU segment boundaries with GPU layer checkpoints during segment replay.

Segment inputs survive the original forward on the selected storage device.
Backward restores one segment, reconstructs its inputs on GPU, and replays one layer at a
time with the existing weight-streaming lifecycle. This avoids keeping several
layers' weights and internal autograd graphs resident together.
"""

from collections.abc import Callable, Sequence
from contextlib import ExitStack, contextmanager, nullcontext

import torch

from ironcore.parallel.random import (
    snapshot_tensor_parallel_rng_tracker,
    tensor_parallel_rng_rewound_to,
)


class _SegmentCheckpointFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, function, scheduler, layer_idx, spill, activation, *aux_args):
        if any(torch.is_tensor(arg) and arg.requires_grad for arg in aux_args):
            raise ValueError("Offloaded recompute auxiliary inputs must not require gradients")
        ctx.function = function
        ctx.scheduler = scheduler
        ctx.layer_idx = layer_idx
        ctx.spill = spill
        ctx.manage_weights = not getattr(function, "manages_weights", False)
        ctx.device = activation.device
        ctx.shape = activation.shape
        ctx.dtype = activation.dtype
        ctx.autocast_enabled = torch.is_autocast_enabled(activation.device.type)
        ctx.autocast_dtype = torch.get_autocast_dtype(activation.device.type)
        ctx.cpu_rng = torch.get_rng_state()
        ctx.cuda_rng = torch.cuda.get_rng_state(activation.device) if activation.is_cuda else None
        ctx.tp_rng = snapshot_tensor_parallel_rng_tracker()
        ctx.aux_args = tuple(
            arg.detach().cpu() if spill and torch.is_tensor(arg) else arg for arg in aux_args
        )
        if spill:
            scheduler.on_sublayer_forward(layer_idx, 0, activation)
        else:
            ctx.save_for_backward(activation)
        with torch.no_grad():
            return function(activation, *aux_args)

    @staticmethod
    def backward(ctx, *grad_outputs):
        if ctx.manage_weights:
            ctx.scheduler.on_backward_layer_start(ctx.layer_idx)
        if ctx.spill:
            activation = ctx.scheduler.on_sublayer_backward(
                ctx.layer_idx, 0, ctx.shape, ctx.dtype, ctx.device
            )
        else:
            activation = ctx.saved_tensors[0]
        activation = activation.detach().requires_grad_(True)
        aux_args = tuple(
            arg.to(ctx.device) if ctx.spill and torch.is_tensor(arg) else arg
            for arg in ctx.aux_args
        )
        devices = [ctx.device.index] if ctx.device.type == "cuda" else []
        with (
            torch.random.fork_rng(devices=devices),
            tensor_parallel_rng_rewound_to(ctx.tp_rng),
            torch.autocast(ctx.device.type, dtype=ctx.autocast_dtype, enabled=ctx.autocast_enabled),
            torch.enable_grad(),
            getattr(ctx.function, "recompute_context", nullcontext)(),
        ):
            torch.set_rng_state(ctx.cpu_rng)
            if ctx.cuda_rng is not None:
                torch.cuda.set_rng_state(ctx.cuda_rng, device=ctx.device)
            outputs = ctx.function(activation, *aux_args)
        outputs = outputs if isinstance(outputs, tuple) else (outputs,)
        active = [
            (out, grad)
            for out, grad in zip(outputs, grad_outputs, strict=True)
            if torch.is_tensor(out) and out.requires_grad and grad is not None
        ]
        if active:
            torch.autograd.backward(
                tuple(out for out, _ in active), tuple(grad for _, grad in active)
            )
        if ctx.manage_weights:
            ctx.scheduler.on_backward_layer_end(ctx.layer_idx)
        return (None, None, None, None, activation.grad) + (None,) * len(aux_args)


def recompute_streamed_segment(
    functions: Sequence[Callable],
    scheduler,
    start: int,
    activation: torch.Tensor,
    *aux_args,
    offload_input: bool = True,
):
    """Return the final hidden state and any per-layer auxiliary losses."""
    if not activation.requires_grad:
        activation = activation.detach().requires_grad_(True)

    if len(functions) == 1:
        # A retained single-layer boundary needs only the existing one replay.
        # Avoid reconstructing a segment and then replaying that layer again.
        scheduler.on_layer_start(start)
        try:
            return _SegmentCheckpointFn.apply(
                functions[0], scheduler, start, offload_input, activation, *aux_args
            )
        finally:
            scheduler.on_layer_end(start)

    def run(hidden, *args):
        losses = []
        replay = torch.is_grad_enabled()
        previous_end = getattr(scheduler, "_recompute_forward_end", None)
        if replay:
            # Do not prefetch a later segment that already finished backward.
            # Its unused staging buffers would survive into the next update.
            scheduler._recompute_forward_end = start + len(functions)
        try:
            for offset, function in enumerate(functions):
                index = start + offset
                scheduler.on_layer_start(index)
                if replay:
                    output = _SegmentCheckpointFn.apply(
                        function, scheduler, index, False, hidden, *args
                    )
                else:
                    output = function(hidden, *args)
                if isinstance(output, tuple):
                    hidden, *aux_losses = output
                    losses.extend(aux_losses)
                else:
                    hidden = output
                scheduler.on_layer_end(index)
        finally:
            if replay:
                scheduler._recompute_forward_end = previous_end
        return (hidden, *losses) if losses else hidden

    @contextmanager
    def replay_context():
        with ExitStack() as stack:
            for function in functions:
                stack.enter_context(getattr(function, "recompute_context", nullcontext)())
            yield

    run.recompute_context = replay_context
    run.manages_weights = True

    return _SegmentCheckpointFn.apply(run, scheduler, start, offload_input, activation, *aux_args)
