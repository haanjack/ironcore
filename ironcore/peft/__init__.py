# Copyright (c) 2025-2026 Jaegeun Han
#
# SPDX-License-Identifier: MIT
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the above copyright notice,
# this list of conditions, and the following disclaimer are retained.
#
# Full license text is available at LICENSE file.

"""
Parameter-Efficient Fine-Tuning (PEFT) module.

Provides implementations of PEFT methods like LoRA for efficient fine-tuning
of large language models with minimal trainable parameters.
"""

from .lora import (
    LoRAColumnParallelLinear,
    LoRAConcatenatedColumnParallel,
    LoRALinear,
    LoRARowParallelLinear,
)
from .utils import (
    count_lora_parameters,
    get_lora_target_modules,
    merge_lora_weights,
    wrap_with_lora_if_target,
)

__all__ = [
    "LoRALinear",
    "LoRAColumnParallelLinear",
    "LoRARowParallelLinear",
    "LoRAConcatenatedColumnParallel",
    "wrap_with_lora_if_target",
    "get_lora_target_modules",
    "count_lora_parameters",
    "merge_lora_weights",
]
