# Copyright (c) 2025-2026 Jaegeun Han
#
# SPDX-License-Identifier: Apache-2.0

from ironcore.config import MainConfig

from .fused_layer_norm import LayerNorm
from .fused_rms_norm import RmsNorm


def get_norm(config: MainConfig):
    """Returns the normalization layer."""
    ln_type = config.model.ln_type.lower()

    if ln_type == "layernorm":
        return LayerNorm(config)
    if ln_type == "rmsnorm":
        return RmsNorm(config)

    raise NotImplementedError(f"{config.ln_type} is not supported")
