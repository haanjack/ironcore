# Copyright (c) 2025-2026 Jaegeun Han
#
# SPDX-License-Identifier: Apache-2.0
"""Checkpoint introspection — inspect and compare checkpoint contents."""

from __future__ import annotations

import pathlib
from pathlib import Path
from typing import Any


def _register_checkpoint_safe_globals() -> None:
    """Register safe globals so weights_only=True can unpickle the dataclass
    configs and pathlib.Path objects stored inside native checkpoints.

    Matches the registration in ironcore.checkpointing.native.load_checkpoint.
    Without this, `ironcore inspect-checkpoint` raises UnpicklingError on
    checkpoints this project writes. (Fable issue #77.)
    """
    import torch

    from ironcore.config import (
        AlignmentConfig,
        DataConfig,
        InitConfig,
        LoRAConfig,
        MainConfig,
        ModelConfig,
        OffloadConfig,
        OperationConfig,
        OptimConfig,
        ParallelConfig,
        PEFTConfig,
        PositionalEmbeddingConfig,
        ProfilerConfig,
        TrainerConfig,
        UtilsConfig,
    )

    torch.serialization.add_safe_globals(
        [
            MainConfig,
            ModelConfig,
            InitConfig,
            OptimConfig,
            DataConfig,
            ParallelConfig,
            TrainerConfig,
            OperationConfig,
            UtilsConfig,
            ProfilerConfig,
            PositionalEmbeddingConfig,
            PEFTConfig,
            LoRAConfig,
            AlignmentConfig,
            OffloadConfig,
            pathlib.PosixPath,
            pathlib.WindowsPath,
            pathlib.PurePosixPath,
            pathlib.PureWindowsPath,
        ]
    )


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

    # Use pre-computed shard totals when available (multi-shard TP checkpoints)
    if "_shard_total_params" in info:
        total_params = info.pop("_shard_total_params")
        dtype_params = info.pop("_shard_dtype_params")
    else:
        total_params = sum(t.numel() for t in state_dict.values())
        dtype_params: dict[str, int] = {}
        for tensor in state_dict.values():
            dt = str(tensor.dtype)
            dtype_params[dt] = dtype_params.get(dt, 0) + tensor.numel()

    info["total_params"] = total_params
    info["total_params_human"] = (
        f"{total_params / 1e9:.2f}B" if total_params >= 1e9 else f"{total_params / 1e6:.1f}M"
    )
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

    return info


def _load_state_dict(checkpoint_path: Path, info: dict) -> dict:
    """Load state dict from HF or native checkpoint, populating *info*.

    For native distributed checkpoints (tp0, tp1, ...), loads all TP shards
    and sums numel across shards for an accurate total parameter count.
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
    if checkpoint_path.is_file():
        import torch

        _register_checkpoint_safe_globals()
        info["format"] = "native"
        state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        return state_dict.get("model_state_dict", state_dict)

    latest_file = checkpoint_path / "latest_step.txt"
    if latest_file.exists():
        import torch

        _register_checkpoint_safe_globals()
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
    """Load all TP shards and sum numel for accurate parameter counting.

    In TP checkpoints, sharded weights (e.g. ColumnParallelLinear) share
    the same key across shards but contain different tensor slices.  To get
    an accurate total parameter count we must sum ``numel()`` of every
    tensor across every shard, not just keep the first occurrence.
    """
    import torch

    info["tp_shards"] = len(tp_shards)
    total_param_count = 0
    dtype_param_counts: dict[str, int] = {}
    training_meta: dict[str, Any] = {}

    for shard_path in tp_shards:
        checkpoint = torch.load(shard_path, map_location="cpu", weights_only=True)
        if "model_state_dict" not in checkpoint:
            raise KeyError(f"Checkpoint at {shard_path} is missing 'model_state_dict' key.")
        if not training_meta:
            training_meta["training_step"] = checkpoint.get("step", "unknown")
            training_meta["training_loss"] = checkpoint.get("loss", "unknown")

        for tensor in checkpoint["model_state_dict"].values():
            total_param_count += tensor.numel()
            dt = str(tensor.dtype)
            dtype_param_counts[dt] = dtype_param_counts.get(dt, 0) + tensor.numel()

        del checkpoint

    # Return a representative state dict from shard 0 for verbose/diff use,
    # but override total_params with the summed count.
    shard0 = torch.load(tp_shards[0], map_location="cpu", weights_only=True)
    state_dict = shard0["model_state_dict"]
    del shard0

    info.update(training_meta)
    # Store accurate counts — inspect_checkpoint() will use these instead
    # of recomputing from the single-shard state_dict.
    info["_shard_total_params"] = total_param_count
    info["_shard_dtype_params"] = dtype_param_counts

    return state_dict


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
