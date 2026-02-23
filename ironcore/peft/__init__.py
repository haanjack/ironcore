# Copyright (c) 2025-2026 Jaegeun Han
#
# SPDX-License-Identifier: Apache-2.0

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
