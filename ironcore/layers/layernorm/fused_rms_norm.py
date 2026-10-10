# Copyright (c) 2025-2026 Jaegeun Han
#
# SPDX-License-Identifier: Apache-2.0

import torch
from torch import nn

from ironcore.layers.module import BaseModule


class RmsNorm(BaseModule):
    """HF-compatible Llama rounding with a frozen-scale CUDA fused path."""

    def __init__(self, config):
        super().__init__(config)

        self.layernorm = nn.RMSNorm(config.model.d_model, eps=float(config.model.ln_eps))

    def forward(self, x):
        if self.config.model.hf_model_type == "llama":
            from .rms_norm_kernel import frozen_scale_rms_norm

            result = frozen_scale_rms_norm(
                x, self.layernorm.weight, self.layernorm.eps, round_before_scale=True
            )
            if result is not None:
                return result
            # Llama rounds the normalized values to the input dtype before
            # applying the scale. nn.RMSNorm is autocast to FP32 on CUDA,
            # which changes this checkpoint's low-precision forward path.
            values = x.float() if x.dtype in (torch.float16, torch.bfloat16) else x
            values = values * torch.rsqrt(
                values.square().mean(-1, keepdim=True) + self.layernorm.eps
            )
            return self.layernorm.weight * values.to(x.dtype)
        return self.layernorm(x)
