# Copyright (c) 2025-2026 Jaegeun Han
#
# SPDX-License-Identifier: Apache-2.0

"""
Checkpointing utilities for ironcore.

This module provides:
- Native checkpoint save/load for training (load_checkpoint, save_checkpoint)
- HuggingFace checkpoint import/export for interoperability (load_from_huggingface, export_to_huggingface)
"""

# Native checkpointing (training resume)
# HuggingFace interop (import/export)
from ironcore.checkpointing.hf_interop import (
    detect_bias_from_hf_state_dict,
    detect_checkpoint_format,
    export_to_huggingface,
    load_from_huggingface,
    load_hf_config,
)
from ironcore.checkpointing.inspect import inspect_checkpoint
from ironcore.checkpointing.native import (
    HFConfigManager,
    load_checkpoint,
    save_checkpoint,
)

# Weight mapping utilities
from ironcore.checkpointing.weight_mapping import (
    Architecture,
    WeightMapper,
    get_architecture,
)

__all__ = [
    # Native checkpointing
    "load_checkpoint",
    "save_checkpoint",
    "HFConfigManager",
    # HuggingFace interop
    "load_from_huggingface",
    "export_to_huggingface",
    "detect_checkpoint_format",
    "detect_bias_from_hf_state_dict",
    "load_hf_config",
    # Weight mapping
    "WeightMapper",
    "Architecture",
    "get_architecture",
    # Inspection
    "inspect_checkpoint",
]
