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

from ironcore.config import LoRAConfig

# Module name mapping from config to actual layer names
MODULE_NAME_MAPPING = {
    "q_proj": "linear_q",
    "k_proj": "linear_kv",  # K and V share the same layer
    "v_proj": "linear_kv",
    "o_proj": "attn_output",
    "up_proj": "up_proj",
    "gate_proj": "up_proj",  # Gate and up share the same layer in some architectures
    "down_proj": "down_proj",
}


def wrap_with_lora_if_target(
    layer,
    module_names: str | list[str],
    lora_config: LoRAConfig,
    concatenated: bool = False,
):
    """
    Wrap a layer with LoRA if it's in the target modules.
    """
    # Lazy imports to avoid circular dependency
    from ironcore.parallel.tensor_parallel.layers import (
        ColumnParallelLinear,
        RowParallelLinear,
    )

    from .lora import (
        LoRAColumnParallelLinear,
        LoRAConcatenatedColumnParallel,
        LoRARowParallelLinear,
    )

    if isinstance(module_names, str):
        module_names = [module_names]

    # Check if any of these modules should have LoRA applied
    is_target = any(name in lora_config.target_modules for name in module_names)

    if not is_target:
        return layer

    # Determine the appropriate LoRA wrapper based on layer type
    if isinstance(layer, ColumnParallelLinear):
        if concatenated and layer.concatenated_weights > 1:
            # Special handling for concatenated weights (K+V, Gate/Up)
            return LoRAConcatenatedColumnParallel(
                layer,
                lora_config,
                target_modules=module_names,
            )
        else:
            return LoRAColumnParallelLinear(layer, lora_config)
    elif isinstance(layer, RowParallelLinear):
        return LoRARowParallelLinear(layer, lora_config)
    else:
        # Not a parallelized layer, return as-is
        return layer


def get_lora_target_modules(target_module_names: list[str]) -> set[str]:
    """
    Convert target module names from config format to actual layer names.

    Args:
        target_module_names: List of module names from config (e.g., ["q_proj", "v_proj"])

    Returns:
        Set of actual layer names in the model
    """
    actual_names = set()
    for name in target_module_names:
        if name in MODULE_NAME_MAPPING:
            actual_names.add(MODULE_NAME_MAPPING[name])
    return actual_names


def freeze_base_model(model, peft_method: str):
    """
    Freeze base model parameters and only keep PEFT adapters trainable.

    Args:
        model: The model to freeze
        peft_method: The PEFT method being used (e.g., "lora")
    """
    if peft_method == "none":
        return

    for name, param in model.named_parameters():
        # Freeze all parameters except PEFT adapters
        if not any(keyword in name for keyword in ["lora_", "adapter_", "prefix_"]):
            param.requires_grad = False


def count_lora_parameters(model) -> tuple[int, int, int]:
    """
    Count LoRA parameters in a model.

    Args:
        model: The model to analyze

    Returns:
        Tuple of (trainable_params, total_params, lora_params)
    """
    trainable_params = 0
    total_params = 0
    lora_params = 0

    for name, param in model.named_parameters():
        num_params = param.numel()
        total_params += num_params

        if param.requires_grad:
            trainable_params += num_params

            # Check if this is a LoRA parameter
            if any(keyword in name for keyword in ["lora_A", "lora_B"]):
                lora_params += num_params

    return trainable_params, total_params, lora_params


def merge_lora_weights(model):
    """
    Merge LoRA weights into base weights for inference.

    This operation is destructive and modifies the base weights in-place.
    After merging, LoRA adapters can be removed for more efficient inference.

    Args:
        model: Model with LoRA layers

    Note:
        This is not yet implemented but provides a hook for future optimization.
    """
    raise NotImplementedError(
        "LoRA weight merging is not yet implemented. "
        "This will be added in a future update for inference optimization."
    )
