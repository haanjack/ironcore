# Copyright (c) 2025-2026 Jaegeun Han
#
# SPDX-License-Identifier: MIT
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the above copyright notice,
# this list of conditions, and the following disclaimer are retained.
#
# Full license text is available at LICENSE file.

# this is place holder for fused rms norm layer


from torch import nn

from ironcore.layers.module import BaseModule


class RmsNorm(BaseModule):
    def __init__(self, config):
        super().__init__(config)

        self.layernorm = nn.RMSNorm(config.model.d_model,
                                    eps=config.model.ln_eps,
                                    bias=not config.model.no_bias)

    def forward(self, x):
        return self.layernorm(x)
