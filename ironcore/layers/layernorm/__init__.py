# Copyright (c) 2025-2026 Jaegeun Han
#
# SPDX-License-Identifier: MIT
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the above copyright notice,
# this list of conditions, and the following disclaimer are retained.
#
# Full license text is available at LICENSE file.

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
