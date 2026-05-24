# Copyright (c) 2025-2026 Jaegeun Han
#
# SPDX-License-Identifier: Apache-2.0
"""CLI wrapper for data preprocessing — delegates to ironcore.preprocess."""

import sys
from pathlib import Path


def register_parser(subparsers) -> None:
    """Register the CLI subcommand arguments."""
    parser = subparsers.add_parser("preprocess", help="Preprocess and/or inspect datasets")
    parser.add_argument(
        "--config", type=str, required=True, help="Path to data configuration YAML file"
    )
    parser.add_argument(
        "--inspect",
        action="store_true",
        help="Run inspection (integrity checks, statistics, packing efficiency) after preprocessing",
    )
    parser.add_argument(
        "--only-inspect",
        action="store_true",
        help="Skip preprocessing and only run inspection on existing files",
    )
    parser.add_argument(
        "--preview",
        type=int,
        default=0,
        help="Number of random samples to preview during inspection (implies --inspect)",
    )


def run_preprocess(args):
    """Run data preprocessing command.

    Delegates serialization to :func:`ironcore.preprocess.preprocess` and
    optionally runs dataset inspection afterwards.

    Args:
        args: Command-line arguments from argparse
            - config: Path to data configuration YAML file
            - inspect: Whether to inspect output files after preprocessing
            - only_inspect: Whether to skip preprocessing and only run inspection
            - preview: Number of samples to preview (implies inspection)
    """
    should_inspect = (
        args.inspect or args.only_inspect or (hasattr(args, "preview") and args.preview > 0)
    )

    if not args.only_inspect:
        config_path = Path(args.config)
        if not config_path.exists():
            print(f"Error: Configuration file not found: {config_path}")
            sys.exit(1)

        from ironcore.preprocess import preprocess

        try:
            preprocess(config_path, verbose=True)
        except (ValueError, FileNotFoundError) as e:
            print(f"\nError during serialization: {e}", file=sys.stderr)
            sys.exit(1)

    if should_inspect:
        print("\nInspecting datasets...")
        from ironcore.cli.inspect import run_inspect

        run_inspect(args)
