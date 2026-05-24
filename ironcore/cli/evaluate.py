# Copyright (c) 2025-2026 Jaegeun Han
#
# SPDX-License-Identifier: Apache-2.0
"""CLI wrapper for evaluation — delegates to ironcore.evaluate."""

import json
import sys
from argparse import Namespace
from pathlib import Path


def register_parser(subparsers) -> None:
    """Register the CLI subcommand arguments."""
    parser = subparsers.add_parser(
        "evaluate", help="Run evaluation benchmarks against a checkpoint"
    )
    parser.add_argument(
        "--config", type=str, required=True, help="Path to training configuration YAML file"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint directory (overrides trainer.model_path)",
    )
    parser.add_argument(
        "--task",
        type=str,
        default="hellaswag",
        help="Evaluation task name (default: hellaswag)",
    )
    parser.add_argument("--num-samples", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file for results JSON",
    )


def run_evaluate(args: Namespace) -> None:
    """Run evaluation benchmarks against a trained checkpoint.

    Thin wrapper that delegates to :func:`ironcore.evaluate.evaluate`.

    Args:
        args: Command-line arguments.
    """
    from ironcore.evaluate import evaluate

    result = evaluate(
        args.config,
        task=args.task,
        checkpoint=args.checkpoint,
        num_samples=args.num_samples,
        batch_size=args.batch_size,
    )

    print()
    print("=" * 50)
    print("Evaluation Results")
    print("=" * 50)

    if result["results"]:
        for metric, value in result["results"].items():
            print(f"  {metric}: {value}")
    else:
        print("  No eval metrics found in output.")

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(result, f, indent=2)
        print(f"\nResults written to: {output_path}")

    if result["status"] != "ok":
        sys.exit(1)
