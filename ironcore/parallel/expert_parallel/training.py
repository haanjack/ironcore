# Copyright (c) 2025-2026 Jaegeun Han
# SPDX-License-Identifier: Apache-2.0
"""Autograd-aware token ownership and bounded EP=2 training support."""

from contextlib import nullcontext

import torch
import torch.distributed as dist
from torch import nn

from .parallel_states import get_expert_model_parallel_group


class _Exchange(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tokens, sends, receives, group):
        ctx.sends, ctx.receives, ctx.group = sends, receives, group
        output = tokens.new_empty(sum(receives), tokens.size(-1))
        dist.all_to_all_single(output, tokens.contiguous(), receives, sends, group=group)
        return output

    @staticmethod
    def backward(ctx, gradient):
        result = gradient.new_empty(sum(ctx.sends), gradient.size(-1))
        dist.all_to_all_single(
            result, gradient.contiguous(), ctx.sends, ctx.receives, group=ctx.group
        )
        return result, None, None, None


def route_expert_tokens(x, indices, weights, experts, expert_start, local_experts, ep_size):
    group = get_expert_model_parallel_group()
    shape = x.shape
    hidden = x.reshape(-1, shape[-1])
    topk = indices.size(-1)
    ids = indices.reshape(-1)
    owners = ids // local_experts
    order = owners.argsort(stable=True)
    send_sizes = torch.bincount(owners, minlength=ep_size).to(torch.int64)
    recv_sizes = torch.empty_like(send_sizes)
    dist.all_to_all_single(recv_sizes, send_sizes, group=group)
    sends, receives = send_sizes.tolist(), recv_sizes.tolist()
    send_tokens = hidden.repeat_interleave(topk, dim=0)[order]
    send_ids = ids[order].contiguous()
    recv_ids = ids.new_empty(sum(receives))
    dist.all_to_all_single(recv_ids, send_ids, receives, sends, group=group)
    recv_tokens = _Exchange.apply(send_tokens, sends, receives, group)
    output = recv_tokens * 0  # keep collective backward even for empty/idle destinations
    for i, expert in enumerate(experts):
        positions = torch.where(recv_ids == expert_start + i)[0]
        if positions.numel():
            expert_output = expert(recv_tokens[positions]).to(output.dtype)
            output = output.index_copy(0, positions, expert_output)
    returned = _Exchange.apply(output, receives, sends, group)
    original = returned[order.argsort()]
    return (
        (original.view(-1, topk, shape[-1]) * weights.reshape(-1, topk, 1)).sum(dim=1).view(shape)
    )


class ExpertParallelModel(nn.Module):
    """Replicated parameters average over DP; owned experts never broadcast."""

    def __init__(self, module, group):
        super().__init__()
        self.module = module
        self.group = group
        self.size = dist.get_world_size(group)
        source = dist.get_global_rank(group, 0)
        for parameter in module.parameters():
            if not getattr(parameter, "is_expert", False):
                dist.broadcast(parameter.data, src=source, group=group)

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)

    def no_sync(self):
        return nullcontext()

    def synchronize_gradients(self):
        for parameter in self.module.parameters():
            if getattr(parameter, "is_expert", False):
                if parameter.grad is not None:
                    parameter.grad.div_(self.size)
            else:
                if parameter.grad is None:
                    parameter.grad = torch.zeros_like(parameter)
                dist.all_reduce(parameter.grad, group=self.group)
                parameter.grad.div_(self.size)
