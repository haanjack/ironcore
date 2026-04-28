# Copyright (c) 2025-2026 Jaegeun Han
#
# SPDX-License-Identifier: Apache-2.0
"""CLI wrapper for checkpoint inspection — delegates to ironcore.checkpointing.inspect."""

import json
import sys
from argparse import Namespace


def register_parser(subparsers) -> None:
    """Register the CLI subcommand arguments."""
    parser = subparsers.add_parser(
        "inspect-checkpoint", help="Inspect checkpoint contents and metadata"
    )
    parser.add_argument("--path", type=str, required=True, help="Path to checkpoint directory")
    parser.add_argument(
        "--compare", type=str, default=None, help="Second checkpoint for weight diff comparison"
    )
    parser.add_argument("--verbose", action="store_true", help="Show per-layer weight stats")
    parser.add_argument("--json", action="store_true", help="Machine-readable JSON output")


def run_inspect_checkpoint(args: Namespace) -> None:
    """Inspect checkpoint contents via :func:`ironcore.checkpointing.inspect.inspect_checkpoint`."""
    from ironcore.checkpointing.inspect import inspect_checkpoint

    try:
        info = inspect_checkpoint(
            args.path,
            verbose=args.verbose,
            compare=args.compare,
        )
    except (FileNotFoundError, ValueError) as e:
        print(f"Error: {e}")
        sys.exit(1)

    if args.json:
        output = {k: v for k, v in info.items() if k != "hf_config"}
        print(json.dumps(output, indent=2, default=str))
        return

    print(f"Checkpoint: {args.path}")
    print(f"Format: {info['format']}")
    if "sharded" in info:
        print(f"Sharded: {info['sharded']}")
    print(f"Total params: {info['total_params_human']} ({info['total_params']:,})")
    print(f"Dtypes: {info['dtype_params']}")
    if "training_step" in info:
        print(f"Training step: {info['training_step']}")
    if "training_loss" in info:
        print(f"Training loss: {info['training_loss']}")
    if "architecture" in info:
        print(f"Architecture: {info['architecture']}")
    if "files" in info:
        print(f"Files: {len(info['files'])}")

    if args.verbose and "layer_stats" in info:
        print(f"\nPer-layer stats ({len(info['layer_stats'])} tensors):")
        for name, stats in sorted(info["layer_stats"].items()):
            print(f"  {name}: {stats['params']:>10,} params, {stats['dtype']}, {stats['shape']}")

    if args.compare and "diffs" in info:
        diffs = info["diffs"]
        only_a = diffs.pop("_only_a", [])
        only_b = diffs.pop("_only_b", [])
        if only_a:
            print(f"\n  Only in first checkpoint: {only_a}")
        if only_b:
            print(f"\n  Only in second checkpoint: {only_b}")
        print(f"\nWeight differences ({len(diffs)} tensors):")
        for name, d in sorted(diffs.items()):
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
        if state_dict_a[name].shape != state_dict_b[name].shape:
            print(
                f"  Skipping {name}: shape mismatch {state_dict_a[name].shape} vs {state_dict_b[name].shape}"
            )
            continue
        # In-place ops on a single float32 copy to reduce peak memory
        a = state_dict_a[name].float()
        b = state_dict_b[name].float()
        a.sub_(b).abs_()
        diffs[name] = {
            "max_abs_diff": a.max().item(),
            "mean_abs_diff": a.mean().item(),
        }

    only_a = set(state_dict_a.keys()) - set(state_dict_b.keys())
    only_b = set(state_dict_b.keys()) - set(state_dict_a.keys())
    diffs["_only_a"] = sorted(only_a)
    diffs["_only_b"] = sorted(only_b)

    return diffs
