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
HuggingFace checkpoint interoperability for ironcore.

This module provides functions to:
- Load pretrained weights from HuggingFace checkpoints
- Export trained models to HuggingFace format

Supported checkpoint formats:
- pytorch_model.bin (single file)
- model.safetensors (single file, recommended)
- Sharded checkpoints (pytorch_model-*.bin or model-*.safetensors)
"""

import json
import re
from pathlib import Path
from typing import Dict, Optional, Union, List

import torch
from torch import nn

from ironcore.checkpointing.weight_mapping import (
    WeightMapper,
    Architecture,
    get_architecture,
)


def detect_checkpoint_format(checkpoint_path: Path) -> Dict:
    """
    Detect the format and files of a HuggingFace checkpoint.

    Args:
        checkpoint_path: Path to the checkpoint directory

    Returns:
        Dict with keys:
            - format: "safetensors" or "pytorch"
            - sharded: bool
            - files: List of checkpoint files
            - index_file: Path to index file (if sharded)
    """
    checkpoint_path = Path(checkpoint_path)

    # Check for safetensors (preferred)
    safetensors_single = checkpoint_path / "model.safetensors"
    safetensors_index = checkpoint_path / "model.safetensors.index.json"

    # Check for pytorch format
    pytorch_single = checkpoint_path / "pytorch_model.bin"
    pytorch_index = checkpoint_path / "pytorch_model.bin.index.json"

    if safetensors_index.exists():
        with open(safetensors_index, "r") as f:
            index = json.load(f)
        files = list(set(index["weight_map"].values()))
        return {
            "format": "safetensors",
            "sharded": True,
            "files": [checkpoint_path / f for f in files],
            "index_file": safetensors_index,
            "weight_map": index["weight_map"],
        }
    elif safetensors_single.exists():
        return {
            "format": "safetensors",
            "sharded": False,
            "files": [safetensors_single],
            "index_file": None,
            "weight_map": None,
        }
    elif pytorch_index.exists():
        with open(pytorch_index, "r") as f:
            index = json.load(f)
        files = list(set(index["weight_map"].values()))
        return {
            "format": "pytorch",
            "sharded": True,
            "files": [checkpoint_path / f for f in files],
            "index_file": pytorch_index,
            "weight_map": index["weight_map"],
        }
    elif pytorch_single.exists():
        return {
            "format": "pytorch",
            "sharded": False,
            "files": [pytorch_single],
            "index_file": None,
            "weight_map": None,
        }
    else:
        raise FileNotFoundError(
            f"No HuggingFace checkpoint found in {checkpoint_path}. "
            "Expected model.safetensors, pytorch_model.bin, or sharded equivalents."
        )


def load_hf_state_dict(checkpoint_path: Path, device: str = "cpu") -> Dict[str, torch.Tensor]:
    """
    Load state dict from HuggingFace checkpoint (handles sharding and safetensors).

    Args:
        checkpoint_path: Path to checkpoint directory
        device: Device to load tensors to

    Returns:
        Merged state dict from all checkpoint files
    """
    ckpt_info = detect_checkpoint_format(checkpoint_path)
    state_dict = {}

    for file_path in ckpt_info["files"]:
        if ckpt_info["format"] == "safetensors":
            try:
                from safetensors.torch import load_file
                file_state_dict = load_file(str(file_path), device=device)
            except ImportError:
                raise ImportError(
                    "safetensors package is required to load .safetensors files. "
                    "Install with: pip install safetensors"
                )
        else:
            file_state_dict = torch.load(
                file_path,
                map_location=device,
                weights_only=True,
            )
        state_dict.update(file_state_dict)

    return state_dict


def load_hf_config(checkpoint_path: Path) -> Dict:
    """Load config.json from HuggingFace checkpoint."""
    config_path = Path(checkpoint_path) / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"config.json not found in {checkpoint_path}")

    with open(config_path, "r") as f:
        return json.load(f)


def load_from_huggingface(
    checkpoint_path: Union[str, Path],
    model: nn.Module,
    architecture: Optional[str] = None,
    strict: bool = False,
    device: Optional[str] = None,
) -> Dict:
    """
    Load weights from a HuggingFace checkpoint into an ironcore model.

    This function:
    1. Auto-detects checkpoint format (safetensors/pytorch, single/sharded)
    2. Auto-detects model architecture from config.json (if not specified)
    3. Maps weight names from HuggingFace to ironcore convention
    4. Handles tensor parallel splitting if needed

    Args:
        checkpoint_path: Path to HuggingFace checkpoint directory
        model: ironcore model to load weights into
        architecture: Model architecture ("gpt2", "llama", etc.). Auto-detected if None.
        strict: If True, raise error for missing/unexpected keys
        device: Device to load weights to. Uses model's device if None.

    Returns:
        Dict with loading info:
            - loaded_keys: Keys that were loaded
            - missing_keys: Keys in model but not in checkpoint
            - unexpected_keys: Keys in checkpoint but not in model

    Example:
        >>> from ironcore.checkpointing import load_from_huggingface
        >>> load_from_huggingface("path/to/llama-7b", model)
    """
    from ironcore.parallel import parallel_states
    from ironcore.parallel.tensor_parallel import comm

    checkpoint_path = Path(checkpoint_path)

    # Determine device
    if device is None:
        device = str(next(model.parameters()).device)

    # Load HuggingFace config to detect architecture
    hf_config = load_hf_config(checkpoint_path)
    if architecture is None:
        architecture = hf_config.get("model_type", "llama")

    # Determine number of layers
    num_layers = hf_config.get(
        "num_hidden_layers",
        hf_config.get("n_layer", 12)  # GPT-2 uses n_layer
    )

    # Create weight mapper
    arch_enum = get_architecture(architecture)
    mapper = WeightMapper(arch_enum, num_layers)

    # Load HuggingFace state dict
    hf_state_dict = load_hf_state_dict(checkpoint_path, device=device)

    # Convert to ironcore format
    ironcore_state_dict = mapper.hf_to_ironcore(hf_state_dict, strict=False)

    # Get model's parameter attributes for tensor parallel
    model_attribs = {
        name: {
            "column_parallel": layer.column_parallel,
            "row_parallel": layer.row_parallel,
            "concatenated_weights": getattr(layer, "concatenated_weights", 1),
        }
        for name, layer in model.named_modules()
        if hasattr(layer, "column_parallel") or hasattr(layer, "row_parallel")
    }

    # Apply tensor parallel splitting if needed
    tp_world_size = parallel_states.get_tensor_model_parallel_world_size()
    if tp_world_size > 1:
        for name in list(ironcore_state_dict.keys()):
            module_name = ".".join(name.split(".")[:-1])
            if module_name in model_attribs:
                attribs = model_attribs[module_name]
                if attribs["column_parallel"]:
                    ironcore_state_dict[name] = comm.split_to_model_parallel_workers(
                        ironcore_state_dict[name], attribs
                    )
                elif attribs["row_parallel"] and "weight" in name:
                    ironcore_state_dict[name] = comm.split_to_model_parallel_workers(
                        ironcore_state_dict[name], attribs
                    )

    # Reshape tensors to match model parameters
    model_state = model.state_dict()
    final_state_dict = {}
    missing_keys = []
    unexpected_keys = list(ironcore_state_dict.keys())

    for name, param in model_state.items():
        if name in ironcore_state_dict:
            loaded = ironcore_state_dict[name]
            if loaded.numel() == param.numel():
                final_state_dict[name] = loaded.reshape_as(param)
            else:
                # Size mismatch - might be due to vocab size differences
                if "embedding" in name or "output_layer" in name:
                    # Handle vocab size mismatch by truncating or padding
                    if loaded.shape[0] > param.shape[0]:
                        final_state_dict[name] = loaded[:param.shape[0]]
                    else:
                        padded = torch.zeros_like(param)
                        padded[:loaded.shape[0]] = loaded
                        final_state_dict[name] = padded
                else:
                    raise ValueError(
                        f"Shape mismatch for {name}: "
                        f"checkpoint has {loaded.shape}, model expects {param.shape}"
                    )
            unexpected_keys.remove(name)
        else:
            missing_keys.append(name)
            # Use model's initialized weights
            final_state_dict[name] = param

    # Load the state dict
    model.load_state_dict(final_state_dict, strict=False)

    result = {
        "loaded_keys": list(set(model_state.keys()) - set(missing_keys)),
        "missing_keys": missing_keys,
        "unexpected_keys": unexpected_keys,
        "architecture": architecture,
        "num_layers": num_layers,
    }

    if strict and (missing_keys or unexpected_keys):
        raise ValueError(
            f"Strict loading failed.\n"
            f"Missing keys: {missing_keys}\n"
            f"Unexpected keys: {unexpected_keys}"
        )

    return result


def export_to_huggingface(
    model: nn.Module,
    output_path: Union[str, Path],
    architecture: str = "llama",
    config: Optional[Dict] = None,
    use_safetensors: bool = True,
    shard_size: Optional[int] = None,
) -> Dict:
    """
    Export an ironcore model to HuggingFace checkpoint format.

    This function:
    1. Gathers weights from tensor parallel workers (if applicable)
    2. Maps weight names from ironcore to HuggingFace convention
    3. Saves in safetensors or pytorch format
    4. Writes config.json

    Args:
        model: ironcore model to export
        output_path: Directory to save the checkpoint
        architecture: Target architecture ("gpt2", "llama", etc.)
        config: HuggingFace config dict. Auto-generated if None.
        use_safetensors: Use safetensors format (recommended)
        shard_size: Max size per shard in bytes. None for single file.

    Returns:
        Dict with export info:
            - files: List of saved files
            - config_file: Path to config.json

    Example:
        >>> from ironcore.checkpointing import export_to_huggingface
        >>> export_to_huggingface(model, "output/my-model", architecture="llama")
    """
    from ironcore.parallel import parallel_states
    from ironcore.parallel.tensor_parallel import comm

    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    # Only rank 0 should save
    if parallel_states.get_data_parallel_group_rank() != 0:
        return {"files": [], "config_file": None}

    # Get model attributes for tensor parallel
    model_attribs = {
        name: {
            "column_parallel": layer.column_parallel,
            "row_parallel": layer.row_parallel,
            "concatenated_weights": getattr(layer, "concatenated_weights", 1),
        }
        for name, layer in model.named_modules()
        if hasattr(layer, "column_parallel") or hasattr(layer, "row_parallel")
    }

    # Gather weights from tensor parallel workers
    ironcore_state_dict = {}
    tp_world_size = parallel_states.get_tensor_model_parallel_world_size()

    for name, param in model.state_dict().items():
        module_name = ".".join(name.split(".")[:-1])
        output_param = param

        if tp_world_size > 1 and module_name in model_attribs:
            attribs = model_attribs[module_name]
            if attribs["column_parallel"]:
                output_param = comm.gather_from_model_parallel_workers(param, attribs)
            elif attribs["row_parallel"] and "weight" in name:
                output_param = comm.gather_from_model_parallel_workers(param, attribs)

        ironcore_state_dict[name] = output_param

    # Only first tensor parallel rank saves (after gathering)
    if parallel_states.get_tensor_model_parallel_rank() != 0:
        return {"files": [], "config_file": None}

    # Determine number of layers from model
    num_layers = sum(1 for name in ironcore_state_dict if re.match(r"model\.layers\.\d+\.", name))
    num_layers = num_layers // len([k for k in ironcore_state_dict if "layers.0." in k]) if num_layers > 0 else 12

    # Count layers properly
    layer_indices = set()
    for name in ironcore_state_dict:
        match = re.match(r"model\.layers\.(\d+)\.", name)
        if match:
            layer_indices.add(int(match.group(1)))
    num_layers = len(layer_indices) if layer_indices else 12

    # Convert to HuggingFace format
    arch_enum = get_architecture(architecture)
    mapper = WeightMapper(arch_enum, num_layers)
    hf_state_dict = mapper.ironcore_to_hf(ironcore_state_dict, strict=False)

    # Move all tensors to CPU for saving
    hf_state_dict = {k: v.cpu() for k, v in hf_state_dict.items()}

    # Save weights
    saved_files = []

    if use_safetensors:
        try:
            from safetensors.torch import save_file

            if shard_size is None:
                # Single file
                output_file = output_path / "model.safetensors"
                save_file(hf_state_dict, str(output_file))
                saved_files.append(output_file)
            else:
                # Sharded
                saved_files.extend(
                    _save_sharded_safetensors(hf_state_dict, output_path, shard_size)
                )
        except ImportError:
            raise ImportError(
                "safetensors package is required for safetensors export. "
                "Install with: pip install safetensors"
            )
    else:
        if shard_size is None:
            output_file = output_path / "pytorch_model.bin"
            torch.save(hf_state_dict, output_file)
            saved_files.append(output_file)
        else:
            saved_files.extend(
                _save_sharded_pytorch(hf_state_dict, output_path, shard_size)
            )

    # Generate and save config
    if config is None:
        config = _generate_hf_config(model, architecture, num_layers)

    config_file = output_path / "config.json"
    with open(config_file, "w") as f:
        json.dump(config, f, indent=2)

    return {
        "files": saved_files,
        "config_file": config_file,
    }


def _save_sharded_safetensors(
    state_dict: Dict[str, torch.Tensor],
    output_path: Path,
    shard_size: int,
) -> List[Path]:
    """Save state dict as sharded safetensors files."""
    from safetensors.torch import save_file

    shards = []
    current_shard = {}
    current_size = 0
    shard_idx = 1

    weight_map = {}

    for key, tensor in state_dict.items():
        tensor_size = tensor.numel() * tensor.element_size()

        if current_size + tensor_size > shard_size and current_shard:
            # Save current shard
            shard_name = f"model-{shard_idx:05d}-of-TOTAL.safetensors"
            save_file(current_shard, str(output_path / shard_name))
            shards.append(output_path / shard_name)
            shard_idx += 1
            current_shard = {}
            current_size = 0

        current_shard[key] = tensor
        current_size += tensor_size
        weight_map[key] = f"model-{shard_idx:05d}-of-TOTAL.safetensors"

    # Save last shard
    if current_shard:
        shard_name = f"model-{shard_idx:05d}-of-TOTAL.safetensors"
        save_file(current_shard, str(output_path / shard_name))
        shards.append(output_path / shard_name)

    # Update shard names with total count
    total_shards = len(shards)
    for i, shard_path in enumerate(shards):
        new_name = shard_path.name.replace("TOTAL", f"{total_shards:05d}")
        new_path = shard_path.parent / new_name
        shard_path.rename(new_path)
        shards[i] = new_path

    # Update weight map
    for key in weight_map:
        weight_map[key] = weight_map[key].replace("TOTAL", f"{total_shards:05d}")

    # Write index file
    index = {
        "metadata": {"total_size": sum(t.numel() * t.element_size() for t in state_dict.values())},
        "weight_map": weight_map,
    }
    index_file = output_path / "model.safetensors.index.json"
    with open(index_file, "w") as f:
        json.dump(index, f, indent=2)

    return shards


def _save_sharded_pytorch(
    state_dict: Dict[str, torch.Tensor],
    output_path: Path,
    shard_size: int,
) -> List[Path]:
    """Save state dict as sharded pytorch files."""
    shards = []
    current_shard = {}
    current_size = 0
    shard_idx = 1

    weight_map = {}

    for key, tensor in state_dict.items():
        tensor_size = tensor.numel() * tensor.element_size()

        if current_size + tensor_size > shard_size and current_shard:
            shard_name = f"pytorch_model-{shard_idx:05d}-of-TOTAL.bin"
            torch.save(current_shard, output_path / shard_name)
            shards.append(output_path / shard_name)
            shard_idx += 1
            current_shard = {}
            current_size = 0

        current_shard[key] = tensor
        current_size += tensor_size
        weight_map[key] = f"pytorch_model-{shard_idx:05d}-of-TOTAL.bin"

    if current_shard:
        shard_name = f"pytorch_model-{shard_idx:05d}-of-TOTAL.bin"
        torch.save(current_shard, output_path / shard_name)
        shards.append(output_path / shard_name)

    total_shards = len(shards)
    for i, shard_path in enumerate(shards):
        new_name = shard_path.name.replace("TOTAL", f"{total_shards:05d}")
        new_path = shard_path.parent / new_name
        shard_path.rename(new_path)
        shards[i] = new_path

    for key in weight_map:
        weight_map[key] = weight_map[key].replace("TOTAL", f"{total_shards:05d}")

    index = {
        "metadata": {"total_size": sum(t.numel() * t.element_size() for t in state_dict.values())},
        "weight_map": weight_map,
    }
    index_file = output_path / "pytorch_model.bin.index.json"
    with open(index_file, "w") as f:
        json.dump(index, f, indent=2)

    return shards


def _generate_hf_config(
    model: nn.Module,
    architecture: str,
    num_layers: int,
) -> Dict:
    """Generate HuggingFace config from ironcore model."""
    config = getattr(model, "config", None)

    # Try to extract dimensions from model
    hidden_size = None
    vocab_size = None
    num_heads = None

    for name, param in model.named_parameters():
        if "embedding.word_embeddings.weight" in name:
            vocab_size, hidden_size = param.shape
            break
        elif "embedding" in name and "weight" in name:
            if len(param.shape) == 2:
                vocab_size, hidden_size = param.shape
                break

    # Try to get num_heads from attention
    for name, module in model.named_modules():
        if hasattr(module, "num_attention_heads"):
            num_heads = module.num_attention_heads
            break

    # Build config based on architecture
    if architecture.lower() in ["gpt2", "gpt"]:
        hf_config = {
            "model_type": "gpt2",
            "architectures": ["GPT2LMHeadModel"],
            "vocab_size": vocab_size or 50257,
            "n_embd": hidden_size or 768,
            "n_layer": num_layers,
            "n_head": num_heads or 12,
            "n_positions": getattr(config, "max_position_embeddings", 1024) if config else 1024,
            "activation_function": "gelu_new",
        }
    else:
        # LLaMA-style config
        hf_config = {
            "model_type": architecture.lower(),
            "architectures": [f"{architecture.title()}ForCausalLM"],
            "vocab_size": vocab_size or 32000,
            "hidden_size": hidden_size or 4096,
            "num_hidden_layers": num_layers,
            "num_attention_heads": num_heads or 32,
            "intermediate_size": (hidden_size or 4096) * 4,
            "max_position_embeddings": getattr(config, "max_position_embeddings", 4096) if config else 4096,
            "rms_norm_eps": 1e-6,
            "tie_word_embeddings": False,
        }

    return hf_config
