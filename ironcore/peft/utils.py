# Copyright (c) 2025-2026 Jaegeun Han
#
# SPDX-License-Identifier: Apache-2.0

import torch

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

    This operation is destructive: it modifies base_layer.weight in-place and
    replaces each LoRA wrapper with its (now-merged) base layer, so a
    subsequent forward pass no longer pays the extra LoRA matmuls. Call this
    on a copy of the model if you still need to keep training the adapters
    afterward.

    Args:
        model: Model with LoRA layers (TP-replicated adapters, per
            docs/peft_guide.md — merging is local to each rank, no
            communication needed since every rank holds the same adapter).

    Returns:
        The same model, mutated in-place, for convenience.
    """
    from .lora import (
        LoRAColumnParallelLinear,
        LoRAConcatenatedColumnParallel,
        LoRARowParallelLinear,
    )

    def _delta(lora) -> torch.Tensor:
        # weight layout is [in_features, out_features] (this layer's forward
        # computes x @ weight); lora_A is [in, r], lora_B is [r, out], so
        # lora_A @ lora_B already matches base_layer.weight's shape.
        return lora.scaling * (lora.lora_A.float() @ lora.lora_B.float())

    @torch.no_grad()
    def _merge_module(module: torch.nn.Module):
        for name, child in list(module.named_children()):
            if isinstance(child, (LoRAColumnParallelLinear, LoRARowParallelLinear)):
                base_weight = child.base_layer.weight.data
                base_weight.add_(_delta(child.lora).to(base_weight.dtype))
                setattr(module, name, child.base_layer)
            elif isinstance(child, LoRAConcatenatedColumnParallel):
                base_weight = child.base_layer.weight.data
                slices = list(torch.split(base_weight, child.output_size_per_concat, dim=1))
                for i, adapter_idx in child.adapter_map.items():
                    adapter = child.lora_adapters[adapter_idx]
                    slices[i] = slices[i] + _delta(adapter).to(base_weight.dtype)
                base_weight.copy_(torch.cat(slices, dim=1))
                setattr(module, name, child.base_layer)
            else:
                _merge_module(child)

    _merge_module(model)
    return model
