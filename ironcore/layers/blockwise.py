# Copyright (c) 2025-2026 Jaegeun Han
# SPDX-License-Identifier: Apache-2.0

"""Token-bounded feed-forward execution with recomputed training intermediates."""

from __future__ import annotations

from collections.abc import Callable

import torch

from ironcore.parallel.random import checkpoint_with_tensor_parallel_rng


def token_chunk_forward(
    function: Callable[[torch.Tensor], torch.Tensor],
    value: torch.Tensor,
    chunk_size: int | None,
    *,
    training: bool,
) -> torch.Tensor:
    """Keep FFN intermediates bounded while retaining the input/output token order.

    Checkpoint each token block independently. A forward-only split would retain
    every block's expanded activations until backward and save little memory.
    Non-reentrant checkpointing also supports trainable weights with frozen inputs.
    The full hidden-size input and output remain resident.
    """
    if chunk_size is None or value.numel() == 0:
        return function(value)
    flat = value.reshape(-1, value.size(-1))
    outputs = []
    for block in flat.split(chunk_size):
        if training and torch.is_grad_enabled():
            output = checkpoint_with_tensor_parallel_rng(function, block, use_reentrant=False)
        else:
            output = function(block)
        outputs.append(output)
    return torch.cat(outputs, dim=0).view_as(value)
