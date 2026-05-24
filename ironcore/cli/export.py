# Copyright (c) 2025-2026 Jaegeun Han
#
# SPDX-License-Identifier: Apache-2.0
"""CLI wrapper for checkpoint export — delegates to ironcore.export."""

import sys
from argparse import Namespace
from pathlib import Path


def register_parser(subparsers) -> None:
    """Register the CLI subcommand arguments."""
    parser = subparsers.add_parser("export", help="Export checkpoint to HuggingFace format")
    parser.add_argument("--config", type=str, required=True, help="Path to training config YAML")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Checkpoint path (overrides trainer.model_path)",
    )
    parser.add_argument(
        "--output-dir", type=str, required=True, help="Output directory for HF checkpoint"
    )
    parser.add_argument(
        "--format",
        type=str,
        default="safetensors",
        choices=["safetensors", "pytorch"],
        help="Output format (default: safetensors)",
    )
    parser.add_argument(
        "--shard-size", type=int, default=None, help="Shard size in MB (no sharding if omitted)"
    )
    parser.add_argument(
        "--architecture",
        type=str,
        default=None,
        help="Target architecture (auto-detect if omitted)",
    )


def run_export(args: Namespace) -> None:
    """Export an IronCore checkpoint to HuggingFace format.

    Thin wrapper that parses CLI arguments and delegates to
    :func:`ironcore.export.export`.

    Args:
        args: Command-line arguments.
    """
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Error: config not found: {config_path}")
        sys.exit(1)

    from ironcore.train import load_full_config

    config = load_full_config(config_path)

    # Convert shard size from MB to bytes
    shard_size: int | None = None
    if args.shard_size:
        shard_size = args.shard_size * 1024 * 1024

    from ironcore.export import export

    result = export(
        config,
        args.output_dir,
        checkpoint=args.checkpoint,
        architecture=args.architecture,
        use_safetensors=(args.format == "safetensors"),
        shard_size=shard_size,
    )

    output_dir = result["output_dir"]
    print(f"Loaded checkpoint from step {result['step']}")
    print(f"Model: {config.model.name} ({result['num_params']:,} params)")
    print(f"\nExported to: {output_dir}")
    for f in result["files"]:
        size_mb = f.stat().st_size / (1024 * 1024) if f.exists() else 0
        print(f"  {f.name} ({size_mb:.1f} MB)")
    print(f"Config: {result['config_file']}")
