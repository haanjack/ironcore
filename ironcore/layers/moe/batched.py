# Copyright (c) 2025-2026 Jaegeun Han
# SPDX-License-Identifier: Apache-2.0
"""Padded batched expert GEMMs with one TP reduction per routed layer."""

import torch

from ironcore.parallel.random import checkpoint_with_tensor_parallel_rng
from ironcore.parallel.tensor_parallel import comm


def batched_experts(
    x: torch.Tensor,
    indices: torch.Tensor,
    weights: torch.Tensor,
    experts: torch.nn.ModuleList,
) -> torch.Tensor:
    """Compute routed experts, optionally bounding per-expert token padding."""
    hidden = x.reshape(-1, x.size(-1))
    topk = indices.size(-1)
    flat = indices.reshape(-1)
    order = flat.argsort(stable=True)
    counts = torch.bincount(flat, minlength=len(experts))
    sizes = counts.tolist()
    capacity = max(sizes)
    ids = flat[order]
    slots = torch.arange(flat.numel(), device=x.device) - (counts.cumsum(0) - counts)[ids]
    repeated = comm.copy_inputs_to_model_parallel_workers(hidden).repeat_interleave(topk, dim=0)[
        order
    ]
    chunk_size = experts[0].config.trainer.mlp_chunk_size

    if capacity == 0:
        return x * 0

    def project(packed):
        return _batched_projection(packed, experts, sizes)

    if chunk_size is None:
        # Preserve the original single-GEMM path when chunking is disabled.
        destinations = ids * capacity + slots
        packed = (
            hidden.new_zeros(len(experts) * capacity, hidden.size(-1))
            .index_copy(0, destinations, repeated)
            .view(len(experts), capacity, hidden.size(-1))
        )
        sorted_outputs = project(packed).reshape(-1, hidden.size(-1))[destinations]
    else:
        sorted_outputs = torch.zeros_like(repeated)
        width = min(capacity, chunk_size)
        for start in range(0, capacity, width):
            block_width = min(width, capacity - start)
            positions = ((slots >= start) & (slots < start + block_width)).nonzero().flatten()
            destinations = ids[positions] * block_width + slots[positions] - start

            def project_tokens(tokens, destinations=destinations, block_width=block_width):
                # Save compact routed inputs, rebuilding padding only for the
                # current GEMM block during forward/backward recomputation.
                packed = (
                    tokens.new_zeros(len(experts) * block_width, tokens.size(-1))
                    .index_copy(0, destinations, tokens)
                    .view(len(experts), block_width, tokens.size(-1))
                )
                return project(packed).reshape(-1, tokens.size(-1))[destinations]

            tokens = repeated[positions]
            if experts[0].training and torch.is_grad_enabled():
                outputs = checkpoint_with_tensor_parallel_rng(
                    project_tokens, tokens, use_reentrant=False
                )
            else:
                outputs = project_tokens(tokens)
            sorted_outputs = sorted_outputs.index_copy(
                0, positions, outputs.to(sorted_outputs.dtype)
            )
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


def _batched_projection(
    packed: torch.Tensor, experts: torch.nn.ModuleList, sizes: list[int]
) -> torch.Tensor:
    """One bounded expert batch; routing and final TP reduction stay outside."""

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
    return torch.bmm(activated, stack("down_proj"))
