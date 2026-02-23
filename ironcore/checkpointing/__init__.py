# Copyright (c) 2025-2026 Jaegeun Han
#
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Checkpointing utilities for ironcore.

This module provides:
- Native checkpoint save/load for training (load_checkpoint, save_checkpoint)
- HuggingFace checkpoint import/export for interoperability (load_from_huggingface, export_to_huggingface)
"""

# Native checkpointing (training resume)
# HuggingFace interop (import/export)
from ironcore.checkpointing.hf_interop import (
    detect_checkpoint_format,
    export_to_huggingface,
    load_from_huggingface,
    load_hf_config,
)
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
    "load_hf_config",
    # Weight mapping
    "WeightMapper",
    "Architecture",
    "get_architecture",
]
