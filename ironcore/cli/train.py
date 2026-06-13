# Copyright (c) 2025-2026 Jaegeun Han
#
# SPDX-License-Identifier: Apache-2.0
"""Training CLI command."""

import sys


def register_parser(subparsers) -> None:
    """Register the CLI subcommand arguments."""
    parser = subparsers.add_parser("train", help="Run training")
    parser.add_argument(
        "--config", type=str, required=True, help="Path to training configuration YAML file"
    )
    parser.add_argument(
        "--micro-batch-size", type=int, default=None, help="Override trainer.micro_batch_size"
    )
    parser.add_argument(
        "--train-batch-size", type=int, default=None, help="Override trainer.train_batch_size"
    )


def run_train(args):
    """Run training command.

    Args:
        args: Command-line arguments from argparse
            - config: Path to training configuration YAML file
    """
    from ironcore.train import load_full_config, train

    try:
        overrides = {}
        if args.micro_batch_size is not None:
            overrides["trainer.micro_batch_size"] = args.micro_batch_size
        if args.train_batch_size is not None:
            overrides["trainer.train_batch_size"] = args.train_batch_size
        config = load_full_config(args.config, overrides=overrides or None)
        train(config)
    except (ValueError, FileNotFoundError) as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
