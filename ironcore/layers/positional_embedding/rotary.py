# Copyright (c) 2025-2026 Jaegeun Han
#
# SPDX-License-Identifier: MIT
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the above copyright notice,
# this list of conditions, and the following disclaimer are retained.
#
# Full license text is available at LICENSE file.

import torch
import torch.nn as nn


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

        theta = 1.0 / \
            (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
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

    def forward(self, x: torch.Tensor):
        # x: [batch_size, seq_len, num_heads, head_dim]
        x = x.transpose(0, 1)
        seq_len = x.size(1)

        if seq_len > self.max_seq_len_cached:
            self._update_rope_cache(seq_len)

        sin_emb = self.sin_emb[:seq_len, :].unsqueeze(
            1).unsqueeze(0).to(x.device)
        cos_emb = self.cos_emb[:seq_len, :].unsqueeze(
            1).unsqueeze(0).to(x.device)

        x = self.apply_rotary_pos_emb(
            x, sin_emb, cos_emb).transpose(0, 1).contiguous()

        return x

    def apply_rotary_pos_emb(
        self, x: torch.Tensor, sin_emb: torch.Tensor, cos_emb: torch.Tensor
    ):
        # x: [batch_size, seq_len, num_heads, head_dim]
        x1 = x[..., ::2]
        x2 = x[..., 1::2]
        sin_emb = sin_emb.repeat(x.size(0), 1, x.size(2), 1)
        cos_emb = cos_emb.repeat(x.size(0), 1, x.size(2), 1)
        x_rotated = torch.stack(
            [x1 * cos_emb - x2 * sin_emb, x1 * sin_emb + x2 * cos_emb], dim=-1
        )
        x_rotated = x_rotated.flatten(-2)
        return x_rotated
