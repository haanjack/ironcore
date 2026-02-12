# Copyright (c) 2025-2026 Jaegeun Han
#
# SPDX-License-Identifier: MIT
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the above copyright notice,
# this list of conditions, and the following disclaimer are retained.
#
# Full license text is available at LICENSE file.

from .dummy import DummyModel
from .transformer import TransformerModel

SUPPORTED_TRANSFORMER_PREFIXES = ["GPT", "LLAMA", "GEMMA1", "QWEN", "PHI1", "PHI2"]
SUPPORTED_VLA_PREFIXES = ["VLA", "VLA-", "ROBOT"]


def get_model_provider_func(config):
    """Get model class based on configuration.

    Args:
        config: MainConfig with model.name specifying architecture

    Returns:
        Model class to instantiate
    """
    model_name = config.model.name.upper()

    if model_name == "DUMMY":
        return DummyModel

    # Check for VLA models
    if any(model_name.startswith(prefix) for prefix in SUPPORTED_VLA_PREFIXES):
        from .vla_model import VLAModel

        return VLAModel

    # Standard transformer models
    if any(model_name.startswith(prefix) for prefix in SUPPORTED_TRANSFORMER_PREFIXES):
        return TransformerModel

    raise NotImplementedError(f"Model architecture '{config.model.name}' is not supported")


__all__ = [
    "DummyModel",
    "TransformerModel",
    "get_model_provider_func",
]
