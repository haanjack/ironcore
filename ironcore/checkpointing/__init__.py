# Copyright (c) 2025-2026 Jaegeun Han
#
# SPDX-License-Identifier: MIT
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice,
#    this list of conditions, and the following disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions, and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
# 3. Neither the name of the copyright holder nor the names of its contributors
#    may be used to endorse or promote products derived from this software
#    without specific prior written permission.
#
# Full license text is available at LICENSE file.

"""
Checkpointing utilities for ironcore.

This module provides:
- Native checkpoint save/load for training (load_checkpoint, save_checkpoint)
- HuggingFace checkpoint import/export for interoperability (load_from_huggingface, export_to_huggingface)
"""

# Native checkpointing (training resume)
from ironcore.checkpointing.native import (
    load_checkpoint,
    save_checkpoint,
    HFConfigManager,
)

# HuggingFace interop (import/export)
from ironcore.checkpointing.hf_interop import (
    load_from_huggingface,
    export_to_huggingface,
    detect_checkpoint_format,
    load_hf_config,
)

# Weight mapping utilities
from ironcore.checkpointing.weight_mapping import (
    WeightMapper,
    Architecture,
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
    "load_hf_config",
    # Weight mapping
    "WeightMapper",
    "Architecture",
    "get_architecture",
]
