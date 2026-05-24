# Copyright (c) 2025-2026 Jaegeun Han
#
# SPDX-License-Identifier: Apache-2.0
"""Experiment report generator."""

from argparse import Namespace
from pathlib import Path

from .reports import (
    REPORT_TEMPLATE,
    format_report,
    gather_experiment_metadata,
    write_report,
)


def register_parser(subparsers) -> None:
    """Register the CLI subcommand arguments."""
    parser = subparsers.add_parser("gen-report", help="Generate experiment reports")
    parser.add_argument("--name", type=str, required=True, help="Experiment name")
    parser.add_argument(
        "--category",
        type=str,
        required=True,
        choices=["pretrain", "sft", "dpo", "grpo", "scaling", "parallelism", "mfu", "profile"],
        help="Experiment category",
    )
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--checkpoint-dir", type=str, default=None)
    parser.add_argument("--log-dir", type=str, default=None)
    parser.add_argument(
        "--output-dir",
        type=str,
        default="experiments/",
        help="Output directory for reports",
    )
    parser.add_argument("--status", type=str, default="PENDING")
    parser.add_argument("--objective", type=str, default=None)
    parser.add_argument("--analysis", type=str, default=None)
    parser.add_argument("--conclusion", type=str, default=None)
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Prompt for all fields interactively",
    )


def run_gen_report(args: Namespace) -> None:
    """Generate a markdown experiment report.

    Args:
        args: Command-line arguments.
            - name: Experiment name
            - category: Experiment category
            - config: Optional training config path
            - checkpoint_dir: Optional checkpoint directory
            - log_dir: Optional log directory
            - output_dir: Output directory (default: experiments/)
            - status: Report status (default: PENDING)
            - objective, analysis, conclusion: Free-text fields
            - interactive: Prompt for fields interactively
    """
    # Gather metadata from config
    metadata = gather_experiment_metadata(args.config)

    # Apply CLI overrides
    metadata["name"] = args.name
    metadata["category"] = args.category

    if args.objective:
        metadata["objective"] = args.objective
    if args.analysis:
        metadata["analysis"] = args.analysis
    if args.conclusion:
        metadata["conclusion"] = args.conclusion
    if args.status:
        metadata["status"] = args.status

    if args.checkpoint_dir:
        metadata["checkpoint_path"] = args.checkpoint_dir
    if args.log_dir:
        metadata["log_path"] = args.log_dir

    # Interactive mode
    if args.interactive:
        print(f"Generating report: {args.name}")
        print(f"Category: {args.category}")
        print()

        metadata["objective"] = (
            input("Objective [press Enter to keep existing]: ").strip() or metadata["objective"]
        )
        metadata["analysis"] = (
            input("Analysis [press Enter to keep existing]: ").strip() or metadata["analysis"]
        )
        metadata["conclusion"] = (
            input("Conclusion [press Enter to keep existing]: ").strip() or metadata["conclusion"]
        )
        metadata["status"] = (
            input("Status (PASS/FAIL/PARTIAL/PENDING) [press Enter to keep existing]: ").strip()
            or metadata["status"]
        )
        metadata["criteria"] = (
            input("Pass/fail criteria [press Enter to keep existing]: ").strip()
            or metadata["criteria"]
        )
        metadata["next_steps"] = (
            input("Next steps [press Enter to keep existing]: ").strip() or metadata["next_steps"]
        )

    # Fill report template
    report_content = format_report(REPORT_TEMPLATE, **metadata)

    # Write report
    output_path = Path(args.output_dir) / args.category / f"{args.name}.md"
    written_path = write_report(report_content, output_path)

    print(f"Report generated: {written_path}")
