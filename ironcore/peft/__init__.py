# Copyright (c) 2025-2026 Jaegeun Han
#
# SPDX-License-Identifier: Apache-2.0

"""
Parameter-Efficient Fine-Tuning (PEFT) module.

Provides implementations of PEFT methods like LoRA for efficient fine-tuning
of large language models with minimal trainable parameters.
"""

from .adapter_io import load_lora_adapter, save_lora_adapter
from .lora import (
    LoRAColumnParallelLinear,
    LoRAConcatenatedColumnParallel,
    LoRALinear,
    LoRARowParallelLinear,
)
from .utils import (
    count_lora_parameters,
    merge_lora_weights,
    wrap_with_lora_if_target,
)

__all__ = [
    "load_lora_adapter",
    "save_lora_adapter",
    "LoRALinear",
    "LoRAColumnParallelLinear",
    "LoRARowParallelLinear",
    "LoRAConcatenatedColumnParallel",
    "wrap_with_lora_if_target",
    "count_lora_parameters",
    "merge_lora_weights",
]
