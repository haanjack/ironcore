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
