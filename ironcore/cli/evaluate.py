# Copyright (c) 2025-2026 Jaegeun Han
#
# SPDX-License-Identifier: Apache-2.0
"""Evaluation runner — run benchmarks against trained checkpoints."""

import json
import sys
from argparse import Namespace
from pathlib import Path

from ironcore.utils import deep_merge, load_yaml_config

from .utils import launch_training, write_temp_config


def run_evaluate(args: Namespace) -> None:
    """Run evaluation benchmarks against a trained checkpoint.

    Launches a training subprocess with train_steps=0 and evaluation enabled.
    The BaseTrainer.evaluate() method runs the configured eval tasks.

    Args:
        args: Command-line arguments.
            - config: Path to training config YAML
            - checkpoint: Optional checkpoint path
            - task: Evaluation task name (default: hellaswag)
            - num_samples: Optional number of samples
            - batch_size: Optional batch size
            - output: Optional output file for results JSON
    """
    config_path = Path(args.config)

    if not config_path.exists():
        print(f"Error: Configuration file not found: {config_path}")
        sys.exit(1)

    config = load_yaml_config(config_path)

    task_name = args.task

    print(f"Running evaluation: {task_name}")
    print(f"  Config: {config_path}")

    # Override config for eval-only run
    overrides: dict = {
        "operation": {
            "train_steps": 0,
            "no_save": True,
        },
        "profiler": {
            "gpu_profiler": False,
            "torch_profiler": False,
            "comm_profiler": False,
            "layer_timing": False,
        },
    }

    # Set checkpoint path
    if args.checkpoint:
        overrides["trainer"] = {"model_path": args.checkpoint}

    # Set eval parameters
    if args.num_samples:
        overrides.setdefault("operation", {})["eval_samples"] = args.num_samples
    if args.batch_size:
        overrides.setdefault("trainer", {})["eval_batch_size"] = args.batch_size

    # Ensure evaluation datasets are configured for the task
    # If no eval_datasets exist and task is specified, add it
    data_config = config.get("data", {})
    if isinstance(data_config, dict) and not data_config.get("eval_datasets"):
        data_config["eval_datasets"] = [{"name": task_name}]

    patched = deep_merge(config, overrides)

    tp_size = config.get("trainer", {}).get("tensor_model_parallel_size", 1)
    num_gpus = max(1, tp_size)

    checkpoint_path = args.checkpoint or config.get("trainer", {}).get("model_path", "N/A")
    print(f"  Checkpoint: {checkpoint_path}")
    print(f"  GPUs: {num_gpus}")
    print()

    # Launch
    temp_path = write_temp_config(patched, original_config_path=config_path)
    print("Starting evaluation...")

    try:
        result = launch_training(str(temp_path), num_gpus=num_gpus, timeout=3600)

        stdout = result.stdout
        stderr = result.stderr

        if result.returncode != 0:
            print("Evaluation failed!")
            if stderr:
                print("STDERR (last 30 lines):")
                for line in stderr.strip().split("\n")[-30:]:
                    print(f"  {line}")
            sys.exit(1)

    except Exception as e:
        print(f"Error launching evaluation: {e}")
        sys.exit(1)

    # Parse eval results from stdout
    import re

    # Look for eval metric lines: typically "eval/<task_name>: <value>"
    eval_pattern = r"(eval/\S+):\s*([\d.]+)"
    eval_matches = re.findall(eval_pattern, stdout)

    results = {}
    for metric_name, value in eval_matches:
        results[metric_name] = float(value)

    # Also look for accuracy-style results
    acc_pattern = r"accuracy:\s*([\d.]+)"
    acc_matches = re.findall(acc_pattern, stdout)
    if acc_matches:
        results["accuracy"] = float(acc_matches[-1])

    # Print summary
    print("\n" + "=" * 50)
    print("Evaluation Results")
    print("=" * 50)
    if results:
        for metric, value in results.items():
            print(f"  {metric}: {value}")
    else:
        print("  No eval metrics found in output.")
        print("  Check that eval_datasets are configured in your data config.")

    # Write output
    if args.output:
        output = {
            "task": task_name,
            "config": str(config_path),
            "checkpoint": str(checkpoint_path),
            "results": results,
        }
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(output, f, indent=2)
        print(f"\nResults written to: {output_path}")

    print()
