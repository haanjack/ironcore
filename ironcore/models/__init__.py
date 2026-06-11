# Copyright (c) 2025-2026 Jaegeun Han
#
# SPDX-License-Identifier: Apache-2.0

from .dummy import DummyModel
from .transformer import TransformerModel


def get_model_provider_func(config):
    if config.model.name.lower() == "dummy":
        return DummyModel
    return TransformerModel


__all__ = [
    "DummyModel",
    "TransformerModel",
]
