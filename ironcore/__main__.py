"""
CLI entry point for IronCore.

IronCore: High-Performance Research Platform for LLM Training

Supports subcommands:
    - preprocess: Preprocess and/or inspect datasets
    - train: Run training
"""

import argparse
import sys
from pathlib import Path


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="ironcore",
        description="IronCore: High-Performance Research Platform for LLM Training"
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # ========================================
    # Subcommand: preprocess
    # ========================================
    preprocess_parser = subparsers.add_parser(
        "preprocess",
        help="Preprocess and/or inspect datasets"
    )
    preprocess_parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to data configuration YAML file"
    )
    preprocess_parser.add_argument(
        "--inspect",
        action="store_true",
        help="Run inspection (integrity checks, statistics, packing efficiency) after preprocessing"
    )
    preprocess_parser.add_argument(
        "--only-inspect",
        action="store_true",
        help="Skip preprocessing and only run inspection on existing files"
    )
    preprocess_parser.add_argument(
        "--preview",
        type=int,
        default=0,
        help="Number of random samples to preview during inspection (implies --inspect)"
    )

    # ========================================
    # Subcommand: train
    # ========================================
    train_parser = subparsers.add_parser(
        "train",
        help="Run training"
    )
    train_parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to training configuration YAML file"
    )

    # Parse arguments
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    # Execute command
    if args.command == "preprocess":
        from ironcore.cli.preprocess import run_preprocess
        run_preprocess(args)
    elif args.command == "train":
        from ironcore.cli.train import run_train
        run_train(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
