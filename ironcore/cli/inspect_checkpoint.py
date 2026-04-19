# Copyright (c) 2025-2026 Jaegeun Han
#
# SPDX-License-Identifier: Apache-2.0
"""Checkpoint introspection and comparison."""

import json
import sys
from argparse import Namespace
from pathlib import Path


def run_inspect_checkpoint(args: Namespace) -> None:
    """Inspect checkpoint contents, metadata, and optionally compare two checkpoints.

    Args:
        args: Command-line arguments.
    """
    checkpoint_path = Path(args.path)
    if not checkpoint_path.exists():
        print(f"Error: path not found: {checkpoint_path}")
        sys.exit(1)

    info: dict = {}
    state_dict = {}

    # Try loading as HuggingFace format
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

    except (FileNotFoundError, ValueError):
        # Try native IronCore format
        latest_file = checkpoint_path / "latest_step.txt"
        if latest_file.exists():
            import torch

            info["format"] = "native"
            with open(latest_file) as f:
                step = f.read().strip()
            step_dir = checkpoint_path / f"step_{step}"
            ckpt_file = step_dir / "pytorch_model.bin"
            if not ckpt_file.exists():
                # Check for distributed TP format: step_X/tp0/pytorch_model.bin
                ckpt_file = checkpoint_path / f"step_{step}" / "tp0" / "pytorch_model.bin"
            if not ckpt_file.exists():
                print(f"Error: checkpoint file not found at {checkpoint_path}")
                sys.exit(1)
            checkpoint = torch.load(ckpt_file, map_location="cpu", weights_only=True)
            state_dict = checkpoint.get("model_state_dict", {})
            info["training_step"] = checkpoint.get("step", "unknown")
            info["training_loss"] = checkpoint.get("loss", "unknown")
        else:
            print(f"Error: no recognizable checkpoint at {checkpoint_path}")
            sys.exit(1)

    # Compute statistics
    total_params = sum(t.numel() for t in state_dict.values())
    info["total_params"] = total_params
    info["total_params_human"] = (
        f"{total_params / 1e9:.2f}B" if total_params >= 1e9 else f"{total_params / 1e6:.1f}M"
    )

    # Dtype distribution
    dtype_params: dict[str, int] = {}
    for tensor in state_dict.values():
        dt = str(tensor.dtype)
        dtype_params[dt] = dtype_params.get(dt, 0) + tensor.numel()
    info["dtype_params"] = dtype_params

    # Per-layer stats
    layer_stats = {}
    for name, tensor in state_dict.items():
        layer_stats[name] = {
            "shape": list(tensor.shape),
            "dtype": str(tensor.dtype),
            "params": tensor.numel(),
        }

    # Compare mode
    if args.compare:
        diffs = _compare_checkpoints(state_dict, args.compare)
        info["diffs"] = diffs

    # Output
    if args.json:
        # Filter non-serializable
        output = {k: v for k, v in info.items() if k != "hf_config"}
        print(json.dumps(output, indent=2, default=str))
    else:
        print(f"Checkpoint: {checkpoint_path}")
        print(f"Format: {info['format']}")
        if "sharded" in info:
            print(f"Sharded: {info['sharded']}")
        print(f"Total params: {info['total_params_human']} ({total_params:,})")
        print(f"Dtypes: {dtype_params}")
        if "training_step" in info:
            print(f"Training step: {info['training_step']}")
        if "training_loss" in info:
            print(f"Training loss: {info['training_loss']}")
        if "architecture" in info:
            print(f"Architecture: {info['architecture']}")
        if "files" in info:
            print(f"Files: {len(info['files'])}")

        if args.verbose:
            print(f"\nPer-layer stats ({len(layer_stats)} tensors):")
            for name, stats in sorted(layer_stats.items()):
                print(
                    f"  {name}: {stats['params']:>10,} params, {stats['dtype']}, {stats['shape']}"
                )

        if args.compare and "diffs" in info:
            print(f"\nWeight differences ({len(info['diffs'])} tensors):")
            for name, d in sorted(info["diffs"].items()):
                print(f"  {name}: max={d['max_abs_diff']:.6e}, mean={d['mean_abs_diff']:.6e}")


def _load_state_dict(checkpoint_path: Path) -> dict:
    """Load state dict from HF or native checkpoint format."""
    try:
        from ironcore.checkpointing.hf_interop import load_hf_state_dict

        return load_hf_state_dict(checkpoint_path, device="cpu")
    except (FileNotFoundError, ValueError):
        pass

    # Try native format
    latest_file = checkpoint_path / "latest_step.txt"
    if latest_file.exists():
        import torch

        with open(latest_file) as f:
            step = f.read().strip()
        ckpt_file = checkpoint_path / f"step_{step}" / "pytorch_model.bin"
        if not ckpt_file.exists():
            ckpt_file = checkpoint_path / f"step_{step}" / "tp0" / "pytorch_model.bin"
        if ckpt_file.exists():
            checkpoint = torch.load(ckpt_file, map_location="cpu", weights_only=True)
            return checkpoint.get("model_state_dict", {})

    print(f"Error: no recognizable checkpoint at {checkpoint_path}")
    sys.exit(1)


def _compare_checkpoints(state_dict_a: dict, compare_path: str) -> dict:
    """Compare two checkpoint state dicts, computing per-tensor diffs."""
    compare = Path(compare_path)
    if not compare.exists():
        print(f"Error: compare path not found: {compare}")
        return {}

    state_dict_b = _load_state_dict(compare)

    diffs = {}
    common_keys = set(state_dict_a.keys()) & set(state_dict_b.keys())
    for name in common_keys:
        diff = (state_dict_a[name].float() - state_dict_b[name].float()).abs()
        diffs[name] = {
            "max_abs_diff": diff.max().item(),
            "mean_abs_diff": diff.mean().item(),
        }

    only_a = set(state_dict_a.keys()) - set(state_dict_b.keys())
    only_b = set(state_dict_b.keys()) - set(state_dict_a.keys())
    if only_a:
        print(f"\n  Only in first checkpoint: {sorted(only_a)}")
    if only_b:
        print(f"\n  Only in second checkpoint: {sorted(only_b)}")

    return diffs
