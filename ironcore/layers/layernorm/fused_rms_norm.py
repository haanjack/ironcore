# Copyright (c) 2025-2026 Jaegeun Han
#
# SPDX-License-Identifier: Apache-2.0

# this is place holder for fused rms norm layer


from torch import nn

from ironcore.layers.module import BaseModule


class RmsNorm(BaseModule):
    def __init__(self, config):
        super().__init__(config)

        self.layernorm = nn.RMSNorm(config.model.d_model, eps=float(config.model.ln_eps))

    def forward(self, x):
        return self.layernorm(x)
