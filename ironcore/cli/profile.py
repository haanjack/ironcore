# Copyright (c) 2025-2026 Jaegeun Han
#
# SPDX-License-Identifier: Apache-2.0
"""Training profiler — simplified wrapper around IronCore's profiling system."""

import sys
from argparse import Namespace
from pathlib import Path

from ironcore.utils import deep_merge, load_yaml_config
from ironcore.utils.subprocess import launch_training, write_temp_config


def register_parser(subparsers) -> None:
    """Register the CLI subcommand arguments."""
    parser = subparsers.add_parser("profile", help="Profile training runs")
    parser.add_argument(
        "--config", type=str, required=True, help="Path to training configuration YAML file"
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="quick",
        choices=["quick", "full", "comm", "memory"],
        help="Profile mode (default: quick)",
    )
    parser.add_argument("--start-step", type=int, default=5)
    parser.add_argument("--end-step", type=int, default=7)
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./logs/profile/",
    )
    parser.add_argument("--ranks", type=str, default="0", help="Comma-separated ranks")
    parser.add_argument("--train-steps", type=int, default=None)


PROFILE_PRESETS = {
    "quick": {
        "layer_timing": True,
        "torch_profiler": False,
        "gpu_profiler": False,
        "comm_profiler": False,
        "memory_snapshot": False,
        "export_chrome_trace": False,
        "export_csv": False,
    },
    "full": {
        "layer_timing": True,
        "torch_profiler": True,
        "gpu_profiler": True,
        "comm_profiler": True,
        "memory_snapshot": True,
        "export_chrome_trace": True,
        "export_csv": True,
    },
    "comm": {
        "layer_timing": False,
        "torch_profiler": False,
        "gpu_profiler": False,
        "comm_profiler": True,
        "memory_snapshot": False,
        "export_chrome_trace": False,
        "export_csv": False,
    },
    "memory": {
        "layer_timing": False,
        "torch_profiler": False,
        "gpu_profiler": False,
        "comm_profiler": False,
        "memory_snapshot": True,
        "oom_monitor": True,
        "export_chrome_trace": False,
        "export_csv": False,
    },
}


def run_profile(args: Namespace) -> None:
    """Profile a training run using the selected mode preset.

    Args:
        args: Command-line arguments.
            - config: Path to training config YAML
            - mode: Profile mode (quick, full, comm, memory)
            - start_step, end_step: Profile window
            - output_dir: Output directory
            - ranks: Comma-separated ranks to profile
            - train_steps: Override total training steps
    """
    config_path = Path(args.config)

    if not config_path.exists():
        print(f"Error: Configuration file not found: {config_path}")
        sys.exit(1)

    config = load_yaml_config(config_path)
    preset = PROFILE_PRESETS[args.mode]

    # Parse ranks
    ranks = [int(r.strip()) for r in args.ranks.split(",")]

    # Build profiler config
    train_steps = args.train_steps or (args.end_step + 2)

    profiler_config = {
        **preset,
        "start": args.start_step,
        "end": args.end_step,
        "ranks": ranks,
        "stop_at_end": True,
        "name": f"profile_{args.mode}",
        "output_dir": args.output_dir,
        "wait_steps": 1,
        "warmup_steps": 1,
        "active_steps": args.end_step - args.start_step,
        "repeat": 1,
    }

    overrides = {
        "profiler": profiler_config,
        "operation": {
            "train_steps": train_steps,
            "no_save": True,
        },
    }
    patched = deep_merge(config, overrides)

    tp_size = config.get("trainer", {}).get("tensor_model_parallel_size", 1)
    num_gpus = max(1, tp_size)

    print(f"Profile Mode: {args.mode}")
    print(f"  Window: steps {args.start_step}–{args.end_step}")
    print(f"  Total steps: {train_steps}")
    print(f"  Ranks: {ranks}")
    print(f"  Output: {args.output_dir}")
    print(f"  GPUs: {num_gpus}")
    print()

    # Write patched config and run
    temp_path = write_temp_config(patched, original_config_path=config_path)
    print("Starting profiled training...")

    try:
        result = launch_training(str(temp_path), num_gpus=num_gpus, timeout=1800)

        if result.returncode != 0:
            print("Training failed during profiling!")
            if result.stderr:
                for line in result.stderr.strip().split("\n")[-30:]:
                    print(f"  {line}")
            sys.exit(1)

        print("\nProfiling complete.")
        print(f"  Results in: {args.output_dir}")

    except Exception as e:
        print(f"Error launching profiled training: {e}")
        sys.exit(1)
