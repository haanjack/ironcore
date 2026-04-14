# Copyright (c) 2025-2026 Jaegeun Han
#
# SPDX-License-Identifier: Apache-2.0
"""
CLI entry point for IronCore.

IronCore: High-Performance Research Platform for LLM Training

Supports subcommands:
    - preprocess: Preprocess and/or inspect datasets
    - train: Run training
    - track: Configure logging backends
    - evaluate: Run evaluation benchmarks
    - gen-report: Generate experiment reports
    - profile: Profile training runs
    - verify-parity: Verify parallelism correctness
    - verify-step: Verify single-step training loss
    - analyze-scaling: Run scaling analysis
    - profile-mfu: Profile MFU utilization
"""

import argparse
import sys


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="ironcore", description="IronCore: High-Performance Research Platform for LLM Training"
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # ========================================
    # Subcommand: preprocess
    # ========================================
    preprocess_parser = subparsers.add_parser(
        "preprocess", help="Preprocess and/or inspect datasets"
    )
    preprocess_parser.add_argument(
        "--config", type=str, required=True, help="Path to data configuration YAML file"
    )
    preprocess_parser.add_argument(
        "--inspect",
        action="store_true",
        help="Run inspection (integrity checks, statistics, packing efficiency) after preprocessing",
    )
    preprocess_parser.add_argument(
        "--only-inspect",
        action="store_true",
        help="Skip preprocessing and only run inspection on existing files",
    )
    preprocess_parser.add_argument(
        "--preview",
        type=int,
        default=0,
        help="Number of random samples to preview during inspection (implies --inspect)",
    )

    # ========================================
    # Subcommand: train
    # ========================================
    train_parser = subparsers.add_parser("train", help="Run training")
    train_parser.add_argument(
        "--config", type=str, required=True, help="Path to training configuration YAML file"
    )

    # ========================================
    # Subcommand: track
    # ========================================
    track_parser = subparsers.add_parser(
        "track", help="Configure logging backends (tensorboard, mlflow, wandb)"
    )
    track_parser.add_argument(
        "--config", type=str, required=True, help="Path to training configuration YAML file"
    )
    track_parser.add_argument(
        "--backends", type=str, default=None,
        help="Comma-separated backends to enable (tensorboard,mlflow,wandb). "
             "If omitted, enters interactive mode.",
    )
    track_parser.add_argument("--wandb-project", type=str, default=None)
    track_parser.add_argument("--wandb-entity", type=str, default=None)
    track_parser.add_argument("--wandb-name", type=str, default=None)
    track_parser.add_argument("--mlflow-uri", type=str, default=None)
    track_parser.add_argument("--mlflow-experiment", type=str, default=None)
    track_parser.add_argument("--tensorboard-dir", type=str, default=None)
    track_parser.add_argument(
        "--output", type=str, default=None,
        help="Output path for patched config (default: print utils snippet to stdout)",
    )

    # ========================================
    # Subcommand: evaluate
    # ========================================
    evaluate_parser = subparsers.add_parser(
        "evaluate", help="Run evaluation benchmarks against a checkpoint"
    )
    evaluate_parser.add_argument(
        "--config", type=str, required=True, help="Path to training configuration YAML file"
    )
    evaluate_parser.add_argument(
        "--checkpoint", type=str, default=None,
        help="Path to checkpoint directory (overrides trainer.model_path)",
    )
    evaluate_parser.add_argument(
        "--task", type=str, default="hellaswag",
        help="Evaluation task name (default: hellaswag)",
    )
    evaluate_parser.add_argument("--num-samples", type=int, default=None)
    evaluate_parser.add_argument("--batch-size", type=int, default=None)
    evaluate_parser.add_argument(
        "--output", type=str, default=None, help="Output file for results JSON",
    )

    # ========================================
    # Subcommand: gen-report
    # ========================================
    gen_report_parser = subparsers.add_parser(
        "gen-report", help="Generate experiment reports"
    )
    gen_report_parser.add_argument("--name", type=str, required=True, help="Experiment name")
    gen_report_parser.add_argument(
        "--category", type=str, required=True,
        choices=["pretrain", "sft", "dpo", "grpo", "scaling", "parallelism", "mfu", "profile"],
        help="Experiment category",
    )
    gen_report_parser.add_argument("--config", type=str, default=None)
    gen_report_parser.add_argument("--checkpoint-dir", type=str, default=None)
    gen_report_parser.add_argument("--log-dir", type=str, default=None)
    gen_report_parser.add_argument(
        "--output-dir", type=str, default="experiments/", help="Output directory for reports",
    )
    gen_report_parser.add_argument("--status", type=str, default="PENDING")
    gen_report_parser.add_argument("--objective", type=str, default=None)
    gen_report_parser.add_argument("--analysis", type=str, default=None)
    gen_report_parser.add_argument("--conclusion", type=str, default=None)
    gen_report_parser.add_argument(
        "--interactive", action="store_true", help="Prompt for all fields interactively",
    )

    # ========================================
    # Subcommand: profile
    # ========================================
    profile_parser = subparsers.add_parser(
        "profile", help="Profile training runs"
    )
    profile_parser.add_argument(
        "--config", type=str, required=True, help="Path to training configuration YAML file"
    )
    profile_parser.add_argument(
        "--mode", type=str, default="quick",
        choices=["quick", "full", "comm", "memory"],
        help="Profile mode (default: quick)",
    )
    profile_parser.add_argument("--start-step", type=int, default=5)
    profile_parser.add_argument("--end-step", type=int, default=7)
    profile_parser.add_argument(
        "--output-dir", type=str, default="./logs/profile/",
    )
    profile_parser.add_argument("--ranks", type=str, default="0", help="Comma-separated ranks")
    profile_parser.add_argument("--train-steps", type=int, default=None)

    # ========================================
    # Subcommand: verify-parity
    # ========================================
    verify_parity_parser = subparsers.add_parser(
        "verify-parity", help="Verify parallelism correctness across configurations"
    )
    verify_parity_parser.add_argument(
        "--config", type=str, required=True, help="Base training configuration YAML file"
    )
    verify_parity_parser.add_argument(
        "--mode", type=str, default="tp", choices=["tp", "dp", "fsdp"],
    )
    verify_parity_parser.add_argument("--tp-sizes", type=str, default="1,2")
    verify_parity_parser.add_argument("--num-steps", type=int, default=10)
    verify_parity_parser.add_argument("--tolerance", type=float, default=1e-5)
    verify_parity_parser.add_argument(
        "--output", type=str, default=None, help="Output file for results JSON",
    )
    verify_parity_parser.add_argument("--seed", type=int, default=42)

    # ========================================
    # Subcommand: verify-step
    # ========================================
    verify_step_parser = subparsers.add_parser(
        "verify-step", help="Run single training step for loss verification"
    )
    verify_step_parser.add_argument(
        "--config", type=str, required=True, help="Path to training configuration YAML file"
    )
    verify_step_parser.add_argument(
        "--reference-loss", type=float, default=None,
        help="Expected reference loss value",
    )
    verify_step_parser.add_argument("--tolerance", type=float, default=0.01)
    verify_step_parser.add_argument(
        "--output", type=str, default=None, help="Output file for results JSON",
    )
    verify_step_parser.add_argument(
        "--verbose", action="store_true", help="Print detailed step information",
    )

    # ========================================
    # Subcommand: analyze-scaling
    # ========================================
    analyze_scaling_parser = subparsers.add_parser(
        "analyze-scaling", help="Run scaling analysis across model/batch sizes"
    )
    analyze_scaling_parser.add_argument(
        "--config", type=str, required=True, help="Base training configuration YAML file"
    )
    analyze_scaling_parser.add_argument(
        "--scale-dimension", type=str, default="model",
        choices=["model", "batch", "compute"],
    )
    analyze_scaling_parser.add_argument(
        "--model-sizes", type=str, default=None,
        help="Comma-separated model config names (for model scaling)",
    )
    analyze_scaling_parser.add_argument(
        "--batch-sizes", type=str, default=None,
        help="Comma-separated batch sizes (for batch scaling)",
    )
    analyze_scaling_parser.add_argument("--num-steps", type=int, default=100)
    analyze_scaling_parser.add_argument(
        "--output-dir", type=str, default="experiments/scaling/",
    )
    analyze_scaling_parser.add_argument(
        "--fit-law", action=argparse.BooleanOptionalAction, default=True,
        help="Fit Chinchilla-style scaling law (requires scipy)",
    )
    analyze_scaling_parser.add_argument(
        "--plot", action="store_true", help="Generate scaling law plots (requires matplotlib)",
    )

    # ========================================
    # Subcommand: profile-mfu
    # ========================================
    profile_mfu_parser = subparsers.add_parser(
        "profile-mfu", help="Profile Model FLOP Utilization"
    )
    profile_mfu_parser.add_argument(
        "--config", type=str, required=True, help="Path to training configuration YAML file"
    )
    profile_mfu_parser.add_argument("--warmup-steps", type=int, default=3)
    profile_mfu_parser.add_argument("--measure-steps", type=int, default=5)
    profile_mfu_parser.add_argument(
        "--hardware-peak", type=float, default=35.6,
        help="Hardware peak TFLOPS/s (default: 35.6 for RTX 3090 bf16)",
    )
    profile_mfu_parser.add_argument(
        "--output", type=str, default=None, help="Output file for MFU results JSON",
    )
    profile_mfu_parser.add_argument(
        "--compare", type=str, default=None,
        help="Path to previous MFU results JSON for comparison",
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
    elif args.command == "track":
        from ironcore.cli.track import run_track
        run_track(args)
    elif args.command == "evaluate":
        from ironcore.cli.evaluate import run_evaluate
        run_evaluate(args)
    elif args.command == "gen-report":
        from ironcore.cli.gen_report import run_gen_report
        run_gen_report(args)
    elif args.command == "profile":
        from ironcore.cli.profile import run_profile
        run_profile(args)
    elif args.command == "verify-parity":
        from ironcore.cli.verify_parity import run_verify_parity
        run_verify_parity(args)
    elif args.command == "verify-step":
        from ironcore.cli.verify_step import run_verify_step
        run_verify_step(args)
    elif args.command == "analyze-scaling":
        from ironcore.cli.analyze_scaling import run_analyze_scaling
        run_analyze_scaling(args)
    elif args.command == "profile-mfu":
        from ironcore.cli.profile_mfu import run_profile_mfu
        run_profile_mfu(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
