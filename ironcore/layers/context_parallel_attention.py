# Copyright (c) 2025-2026 Jaegeun Han
# SPDX-License-Identifier: Apache-2.0

"""Causal context attention with bounded KV ring storage and inverse KV gradients."""

from __future__ import annotations

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn

from ironcore.parallel import parallel_states as ps
from ironcore.parallel.context_parallel import gather_context_parallel


class FlashAttentionKernel:
    """Isolate PyTorch's native FlashAttention forward/LSE/backward operator ABI.

    Inputs are B,H,S,D. The backward kernel accepts the complete attention
    output/LSE, so gradients of partial KV blocks share the global normalization.
    Zero dropout avoids kernel-specific RNG replay across a changing CP topology.
    """

    @staticmethod
    def forward(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, causal: bool) -> tuple:
        return torch.ops.aten._scaled_dot_product_flash_attention(
            query, key, value, 0.0, causal, False
        )

    @staticmethod
    def backward(
        gradient: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        output: torch.Tensor,
        lse: torch.Tensor,
        causal: bool,
        metadata: tuple,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        cu_q, cu_k, max_q, max_k, rng_state, unused = metadata
        return torch.ops.aten._scaled_dot_product_flash_attention_backward(
            gradient.contiguous(),
            query,
            key,
            value,
            output,
            lse,
            cu_q,
            cu_k,
            max_q,
            max_k,
            0.0,
            causal,
            rng_state,
            unused,
        )


def _exchange(
    tensors: list[torch.Tensor], group: dist.ProcessGroup, ranks: list[int], rank: int
) -> tuple[list[torch.Tensor], list[dist.Work]]:
    """Start one ring rotation; buffers must remain alive until work completes."""
    incoming = [torch.empty_like(value) for value in tensors]
    operations = []
    for value, received in zip(tensors, incoming, strict=True):
        operations.extend(
            (
                dist.P2POp(dist.isend, value, ranks[(rank + 1) % len(ranks)], group),
                dist.P2POp(dist.irecv, received, ranks[(rank - 1) % len(ranks)], group),
            )
        )
    return incoming, dist.batch_isend_irecv(operations)


class _RingAttention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor):
        group, ranks = ps.get_context_parallel_group(), ps.get_context_parallel_global_ranks()
        rank, size = ps.get_context_parallel_rank(), len(ranks)
        accumulator = torch.zeros_like(query, dtype=torch.float32)
        lse = torch.full(query.shape[:-1], -torch.inf, device=query.device, dtype=torch.float32)
        current = torch.stack((key, value))
        metadata = None
        for step in range(size):
            owner = (rank - step) % size
            next_buffers, work = (
                _exchange([current], group, ranks, rank) if step + 1 < size else (None, [])
            )
            if owner <= rank:
                partial = FlashAttentionKernel.forward(query, current[0], current[1], owner == rank)
                block_output, block_lse = partial[:2]
                metadata = tuple(partial[2:8])
                combined = torch.logaddexp(lse, block_lse)
                accumulator = accumulator * (lse - combined).exp().unsqueeze(
                    -1
                ) + block_output.float() * (block_lse - combined).exp().unsqueeze(-1)
                lse = combined
            for handle in work:
                handle.wait()
            if next_buffers is not None:
                current = next_buffers[0]
        output = accumulator.to(query.dtype)
        ctx.save_for_backward(query, key, value, output, lse)
        ctx.group, ctx.ranks, ctx.rank, ctx.kernel_metadata = group, ranks, rank, metadata
        return output

    @staticmethod
    def backward(ctx, gradient: torch.Tensor):
        query, key, value, output, lse = ctx.saved_tensors
        size, rank = len(ctx.ranks), ctx.rank
        current = torch.stack((key, value))
        kv_gradient = torch.zeros_like(current, dtype=torch.float32)
        query_gradient = torch.zeros_like(query, dtype=torch.float32)
        for step in range(size):
            owner = (rank - step) % size
            if owner <= rank:
                dq, dk, dv = FlashAttentionKernel.backward(
                    gradient,
                    query,
                    current[0],
                    current[1],
                    output,
                    lse,
                    owner == rank,
                    ctx.kernel_metadata,
                )
                query_gradient.add_(dq.float())
                kv_gradient[0].add_(dk.float())
                kv_gradient[1].add_(dv.float())
            # Contributions follow their KV owner. After size rotations each
            # original owner has the sum from all remote query shards.
            tensors = [kv_gradient] if step + 1 == size else [current, kv_gradient]
            received, work = _exchange(tensors, ctx.group, ctx.ranks, rank)
            for handle in work:
                handle.wait()
            if step + 1 == size:
                kv_gradient = received[0]
            else:
                current, kv_gradient = received
        return (
            query_gradient.to(query.dtype),
            kv_gradient[0].to(key.dtype),
            kv_gradient[1].to(value.dtype),
        )


class ContextParallelAttention(nn.Module):
    """Compute local causal Q against the global sequence; only KV is communicated."""

    def __init__(self, backend: str) -> None:
        super().__init__()
        self.backend = backend

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        if self.backend == "ring":
            if query.device.type != "cuda" or query.dtype not in {torch.float16, torch.bfloat16}:
                raise ValueError(
                    "CP ring requires CUDA FP16/BF16 FlashAttention; use sdpa for FP32 reference"
                )
            output = _RingAttention.apply(
                query.transpose(1, 2).contiguous(),
                key.transpose(1, 2).contiguous(),
                value.transpose(1, 2).contiguous(),
            ).transpose(1, 2)
        elif self.backend == "sdpa":
            key = gather_context_parallel(key, reduce_backward=True)
            value = gather_context_parallel(value, reduce_backward=True)
            length = query.size(1)
            positions = ps.get_context_parallel_rank() * length + torch.arange(
                length, device=query.device
            )
            mask = torch.arange(key.size(1), device=query.device)[None, :] <= positions[:, None]
            output = F.scaled_dot_product_attention(
                query.transpose(1, 2),
                key.transpose(1, 2),
                value.transpose(1, 2),
                attn_mask=mask,
                dropout_p=0.0,
                enable_gqa=query.size(2) != key.size(2),
            ).transpose(1, 2)
        else:
            raise ValueError(f"Unsupported context parallel backend: {self.backend}")
        return output.reshape(*output.shape[:2], -1)
