# Copyright (c) 2025-2026 Jaegeun Han
#
# SPDX-License-Identifier: Apache-2.0
"""Checkpoint introspection — inspect and compare checkpoint contents."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def inspect_checkpoint(
    checkpoint_path: str | Path,
    *,
    verbose: bool = False,
    compare: str | Path | None = None,
) -> dict[str, Any]:
    """Inspect a checkpoint and return structured metadata.

    Supports both HuggingFace and native IronCore checkpoint formats.

    Args:
        checkpoint_path: Path to the checkpoint directory or file.
        verbose: Include per-layer tensor statistics.
        compare: Optional second checkpoint path for comparison.

    Returns:
        Dict with keys: format, total_params, dtype_params, layer_stats,
        and optionally training_step, training_loss, architecture, diffs.
    """
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    info: dict[str, Any] = {}
    state_dict = _load_state_dict(checkpoint_path, info)

    total_params = sum(t.numel() for t in state_dict.values())
    info["total_params"] = total_params
    info["total_params_human"] = (
        f"{total_params / 1e9:.2f}B" if total_params >= 1e9 else f"{total_params / 1e6:.1f}M"
    )

    dtype_params: dict[str, int] = {}
    for tensor in state_dict.values():
        dt = str(tensor.dtype)
        dtype_params[dt] = dtype_params.get(dt, 0) + tensor.numel()
    info["dtype_params"] = dtype_params

    if verbose:
        layer_stats = {}
        for name, tensor in state_dict.items():
            layer_stats[name] = {
                "shape": list(tensor.shape),
                "dtype": str(tensor.dtype),
                "params": tensor.numel(),
            }
        info["layer_stats"] = layer_stats

    if compare is not None:
        info["diffs"] = _compare_checkpoints(state_dict, Path(compare))

    # Free state_dict early to reduce peak memory for large models
    del state_dict

    return info


def _load_state_dict(checkpoint_path: Path, info: dict) -> dict:
    """Load state dict from HF or native checkpoint, populating *info*.

    For native distributed checkpoints (tp0, tp1, ...), loads and merges
    all TP shards for an accurate parameter count.
    """
    # Try HuggingFace format first
    try:
        from ironcore.checkpointing.hf_interop import (
            detect_checkpoint_format,
            load_hf_config,
            load_hf_state_dict,
        )

        ckpt_info = detect_checkpoint_format(checkpoint_path)
        info["format"] = ckpt_info["format"]
        info["sharded"] = ckpt_info["sharded"]
        info["files"] = [str(f) for f in ckpt_info["files"]]

        state_dict = load_hf_state_dict(checkpoint_path, device="cpu")

        try:
            hf_config = load_hf_config(checkpoint_path)
            info["architecture"] = hf_config.get("model_type", "unknown")
            info["hf_config"] = hf_config
        except FileNotFoundError:
            pass

        return state_dict

    except (FileNotFoundError, ValueError):
        pass

    # Try native IronCore format
    latest_file = checkpoint_path / "latest_step.txt"
    if latest_file.exists():
        import torch

        info["format"] = "native"
        with open(latest_file) as f:
            step = f.read().strip()
        step_dir = checkpoint_path / f"step_{step}"

        # Check for distributed TP shards (tp0, tp1, ...)
        tp_shards = sorted(step_dir.glob("tp*/pytorch_model.bin"))
        if tp_shards:
            return _load_distributed_shards(tp_shards, info)

        # Fall back to universal checkpoint
        ckpt_file = step_dir / "pytorch_model.bin"
        if not ckpt_file.exists():
            raise FileNotFoundError(f"No checkpoint file found at {checkpoint_path}")

        checkpoint = torch.load(ckpt_file, map_location="cpu", weights_only=True)
        info["training_step"] = checkpoint.get("step", "unknown")
        info["training_loss"] = checkpoint.get("loss", "unknown")
        if "model_state_dict" not in checkpoint:
            raise KeyError(f"Checkpoint at {ckpt_file} is missing 'model_state_dict' key.")
        info["tp_shards"] = 1
        return checkpoint["model_state_dict"]

    raise ValueError(f"No recognizable checkpoint at {checkpoint_path}")


def _load_distributed_shards(tp_shards: list[Path], info: dict) -> dict:
    """Load and merge all TP shards for accurate parameter counting."""
    import torch

    info["tp_shards"] = len(tp_shards)
    merged: dict = {}

    for shard_path in tp_shards:
        checkpoint = torch.load(shard_path, map_location="cpu", weights_only=True)
        if "model_state_dict" not in checkpoint:
            raise KeyError(f"Checkpoint at {shard_path} is missing 'model_state_dict' key.")
        # Store metadata from first shard
        if "training_step" not in info:
            info["training_step"] = checkpoint.get("step", "unknown")
            info["training_loss"] = checkpoint.get("loss", "unknown")
        shard_state = checkpoint["model_state_dict"]
        for key in shard_state:
            if key not in merged:
                merged[key] = shard_state[key]
        del shard_state, checkpoint

    return merged


def _compare_checkpoints(state_dict_a: dict, compare_path: Path) -> dict[str, Any]:
    """Compare two checkpoint state dicts, returning per-tensor diffs."""
    if not compare_path.exists():
        raise FileNotFoundError(f"Compare path not found: {compare_path}")

    state_dict_b_info: dict[str, Any] = {}
    state_dict_b = _load_state_dict(compare_path, state_dict_b_info)

    diffs: dict[str, Any] = {}
    common_keys = set(state_dict_a.keys()) & set(state_dict_b.keys())
    for name in common_keys:
        if state_dict_a[name].shape != state_dict_b[name].shape:
            continue
        a = state_dict_a[name].float()
        b = state_dict_b[name].float()
        diff = (a - b).abs()
        diffs[name] = {
            "max_abs_diff": diff.max().item(),
            "mean_abs_diff": diff.mean().item(),
        }

    diffs["_only_a"] = sorted(set(state_dict_a.keys()) - set(state_dict_b.keys()))
    diffs["_only_b"] = sorted(set(state_dict_b.keys()) - set(state_dict_a.keys()))

    return diffs
