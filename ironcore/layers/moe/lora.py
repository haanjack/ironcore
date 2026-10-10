# Copyright (c) 2025-2026 Jaegeun Han
# SPDX-License-Identifier: Apache-2.0
"""Low-rank expert projections with batched TP gradient communication."""

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from ironcore.parallel.random import tensor_parallel_rng_fork
from ironcore.parallel.tensor_parallel import comm


@dataclass(frozen=True)
class ExpertAdapter:
    name: str
    offset: int
    scaling: float
    dropout: float
    seed: int


def expert_parameters(experts, active):
    """Keep base shards unchanged; communicate active adapter views once per layer.

    Column A and row B need summed TP gradients. Column B and row A need
    gathered gradients. Build those boundaries outside block recomputation so
    collectives run once for each packed adapter, rather than once per tile.
    """
    if getattr(experts[0], "folded_parameter_lora", False):
        from .granitemoe import folded_expert_parameters

        return folded_expert_parameters(experts, active)
    bias = experts[0].up_proj.bias is not None
    stride = 3 if bias else 2
    parameters = [
        [e.up_proj.weight, e.down_proj.weight, *([e.up_proj.bias] if bias else [])]
        for e in (experts[i] for i in active)
    ]
    adapters = []
    for name in ("gate_proj", "up_proj", "down_proj"):
        first = getattr(experts[0], f"lora_{name}", None)
        if first is None:
            continue
        selected = [getattr(experts[i], f"lora_{name}") for i in active]
        a = torch.stack([adapter.lora_A for adapter in selected])
        b = torch.stack([adapter.lora_B for adapter in selected])
        if name == "down_proj":
            a = comm.scatter_input_to_model_parallel_workers(a.transpose(-1, -2)).transpose(-1, -2)
            b = comm.copy_inputs_to_model_parallel_workers(b)
        else:
            a = comm.copy_inputs_to_model_parallel_workers(a)
            b = comm.scatter_input_to_model_parallel_workers(b)
        adapters.append(
            ExpertAdapter(
                name,
                stride,
                first.scaling,
                first.dropout.p if first.training and first.dropout is not None else 0.0,
                experts[0].config.init.seed,
            )
        )
        for row, local_a, local_b in zip(parameters, a, b, strict=True):
            row.extend((local_a, local_b))
        stride += 2
    return tuple(p for row in parameters for p in row), tuple(adapters), stride


def adapter_projection(tokens, parameters, owners, stride, adapter, multiply, compute_dtype):
    a = torch.stack([parameters[stride * i + adapter.offset] for i in owners]).to(compute_dtype)
    b = torch.stack([parameters[stride * i + adapter.offset + 1] for i in owners]).to(compute_dtype)
    hidden = multiply(tokens.to(compute_dtype), a)
    if adapter.dropout:
        # The mask lives in the replicated rank dimension, including row TP's
        # partial low-rank outputs. Linearity lets the layer's final reduction
        # combine row adapters without a collective for every execution tile.
        with tensor_parallel_rng_fork(adapter.seed, hidden.device):
            hidden = F.dropout(hidden, p=adapter.dropout, training=True)
    return adapter.scaling * multiply(hidden, b)
