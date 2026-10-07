# Copyright (c) 2025-2026 Jaegeun Han
# SPDX-License-Identifier: Apache-2.0
"""Padded batched expert GEMMs with one TP reduction per routed layer."""

import torch

from ironcore.parallel.tensor_parallel import comm


def batched_experts(x, indices, weights, experts):
    hidden = x.reshape(-1, x.size(-1))
    topk = indices.size(-1)
    flat = indices.reshape(-1)
    order = flat.argsort(stable=True)
    counts = torch.bincount(flat, minlength=len(experts))
    sizes = counts.tolist()
    capacity = max(sizes)
    ids = flat[order]
    slots = torch.arange(flat.numel(), device=x.device) - (counts.cumsum(0) - counts)[ids]
    destinations = ids * capacity + slots
    repeated = comm.copy_inputs_to_model_parallel_workers(hidden).repeat_interleave(topk, dim=0)[
        order
    ]
    packed = hidden.new_zeros(len(experts) * capacity, hidden.size(-1)).index_copy(
        0, destinations, repeated
    )
    packed = packed.view(len(experts), capacity, hidden.size(-1))

    def stack(attribute):
        return torch.stack(
            [
                getattr(e, attribute).weight if sizes[i] else getattr(e, attribute).weight.detach()
                for i, e in enumerate(experts)
            ]
        )

    up = torch.bmm(packed, stack("up_proj"))
    if experts[0].up_proj.bias is not None:
        up = (
            up
            + torch.stack(
                [
                    e.up_proj.bias if sizes[i] else e.up_proj.bias.detach()
                    for i, e in enumerate(experts)
                ]
            )[:, None, :]
        )
    activated = experts[0].activation(up)
    down = torch.bmm(activated, stack("down_proj"))
    sorted_outputs = down.reshape(-1, hidden.size(-1))[destinations]
    original = sorted_outputs[order.argsort()].view(-1, topk, hidden.size(-1))
    parallel_weights = comm.copy_inputs_to_model_parallel_workers(weights)
    result = (original * parallel_weights.reshape(-1, topk, 1)).sum(1)
    result = comm.reduce_inputs_from_model_parallel_workers(result)
    if experts[0].down_proj.bias is not None:
        biases = torch.stack(
            [
                e.down_proj.bias if sizes[i] else e.down_proj.bias.detach()
                for i, e in enumerate(experts)
            ]
        )
        result = result + (biases[indices.reshape(-1, topk)] * weights.reshape(-1, topk, 1)).sum(1)
    return result.view_as(x)
