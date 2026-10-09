# Copyright (c) 2025-2026 Jaegeun Han
# SPDX-License-Identifier: Apache-2.0
"""Standalone, unmerged IronCore LoRA adapter weights with an explicit loader."""

from __future__ import annotations

import json
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

from .lora import LoRALinear


def _adapter_parameters(model: torch.nn.Module) -> dict[str, torch.nn.Parameter]:
    ids = {
        id(p)
        for module in model.modules()
        if isinstance(module, LoRALinear)
        for p in module.parameters(recurse=False)
    }
    return {name: p for name, p in model.named_parameters() if id(p) in ids}


def save_lora_adapter(model: torch.nn.Module, directory: str | Path) -> None:
    """Write full replicated adapters on one designated writer, without base weights."""
    parameters = _adapter_parameters(model)
    if not parameters:
        raise ValueError("Model has no LoRA adapters")
    path = Path(directory)
    path.mkdir(parents=True, exist_ok=True)
    config = model.config.peft.lora
    metadata = {
        "format": "ironcore_lora_v1",
        "base_model_name_or_path": model.config.trainer.load_from_hf,
        "r": config.r,
        "alpha": config.alpha,
        "dropout": config.dropout,
        "target_modules": config.target_modules,
        "layout": "A[in_features, rank], B[rank, out_features]; full TP replicas",
    }
    save_file(
        {name: p.detach().cpu().contiguous() for name, p in parameters.items()},
        str(path / "adapter_model.safetensors"),
    )
    (path / "adapter_config.json").write_text(json.dumps(metadata, indent=2) + "\n")


def load_lora_adapter(model: torch.nn.Module, directory: str | Path) -> None:
    """Load adapters after the matching pretrained base, checking every name/shape."""
    path = Path(directory)
    metadata = json.loads((path / "adapter_config.json").read_text())
    config = model.config.peft.lora
    if (
        metadata.get("format") != "ironcore_lora_v1"
        or metadata["r"] != config.r
        or metadata["alpha"] != config.alpha
        or set(metadata["target_modules"]) != set(config.target_modules)
    ):
        raise ValueError("LoRA adapter configuration mismatch")
    parameters = _adapter_parameters(model)
    weights = load_file(str(path / "adapter_model.safetensors"))
    if weights.keys() != parameters.keys():
        raise ValueError("LoRA adapter names do not match the model")
    for name, parameter in parameters.items():
        if weights[name].shape != parameter.shape:
            raise ValueError(f"LoRA adapter shape mismatch for {name}")
    with torch.no_grad():
        for name, parameter in parameters.items():
            parameter.copy_(weights[name])
