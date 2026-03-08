# Copyright (c) 2025-2026 Jaegeun Han
#
# SPDX-License-Identifier: Apache-2.0

import torch
from torch import nn


class RotaryPositionalEmbedding(nn.Module):
    def __init__(
        self,
        head_dim: int,
        max_seq_len: int = 4096,
        base: int = 10000,
        scale: float = 1.0,
        offset: int = 0,
    ):
        super().__init__()
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.rope_base = base
        self.offset = offset
        self.scale = scale

        theta = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer("theta", theta)
        self._update_rope_cache(max_seq_len)

    def _update_rope_cache(self, max_seq_len):
        # create position indexes
        self.max_seq_len_cached = max_seq_len
        position = torch.arange(
            self.offset,
            self.offset + self.max_seq_len_cached,
            dtype=torch.float32,
            device=self.theta.device,
        )

        position *= self.scale

        idx_theta = torch.einsum("i,j->ij", position, self.theta)
        # cache sin and cos
        self.register_buffer("sin_emb", torch.sin(idx_theta), persistent=False)
        self.register_buffer("cos_emb", torch.cos(idx_theta), persistent=False)

    def forward(self, x: torch.Tensor, position_ids: torch.Tensor | None = None):
        # x: [batch_size, seq_len, num_heads, head_dim]
        batch_size, seq_len = x.shape[0], x.shape[1]

        if position_ids is None:
            # Fallback: assume sequential positions starting from 0
            position_ids = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch_size, seq_len)

        max_pos = position_ids.max().item()
        if max_pos >= self.max_seq_len_cached:
            self._update_rope_cache(int(max_pos) + 1)

        # Index into sin/cos using position_ids: [batch, seq_len, head_dim//2]
        # and add a dimension for broadcasting across heads
        sin_emb = self.sin_emb[position_ids].unsqueeze(2).to(x.dtype)
        cos_emb = self.cos_emb[position_ids].unsqueeze(2).to(x.dtype)

        x = self.apply_rotary_pos_emb(x, sin_emb, cos_emb)

        return x

    def apply_rotary_pos_emb(self, x: torch.Tensor, sin_emb: torch.Tensor, cos_emb: torch.Tensor):
        # x: [batch_size, seq_len, num_heads, head_dim]
        # sin_emb/cos_emb: [batch, seq_len, 1, head_dim//2]
        x1 = x[..., ::2]
        x2 = x[..., 1::2]
        
        # RoPE rotation formula: [x1*cos - x2*sin, x1*sin + x2*cos]
        x_rotated = torch.stack([x1 * cos_emb - x2 * sin_emb, x1 * sin_emb + x2 * cos_emb], dim=-1)
        x_rotated = x_rotated.flatten(-2)
        return x_rotated
