# Copyright (c) 2025-2026 Jaegeun Han
#
# SPDX-License-Identifier: Apache-2.0
"""Single-step training loss verification."""

import json
import sys
from argparse import Namespace
from pathlib import Path

from ironcore.utils import deep_merge, load_yaml_config

from .utils import launch_training, parse_metrics_from_stdout, write_temp_config


def run_verify_step(args: Namespace) -> None:
    """Run exactly 1 training step and report loss.

    Args:
        args: Command-line arguments.
            - config: Path to training config YAML
            - reference_loss: Optional expected loss value
            - tolerance: Tolerance for comparison (default: 0.01)
            - output: Optional output file for results JSON
            - verbose: Print detailed info
    """
    config_path = Path(args.config)

    if not config_path.exists():
        print(f"Error: Configuration file not found: {config_path}")
        sys.exit(1)

    config = load_yaml_config(config_path)

    # Override for 1-step run
    overrides = {
        "operation": {
            "train_steps": 1,
            "no_save": True,
        },
        "trainer": {
            "log_interval": 1,
        },
        "profiler": {
            "gpu_profiler": False,
            "torch_profiler": False,
            "comm_profiler": False,
            "layer_timing": False,
        },
    }
    patched = deep_merge(config, overrides)

    # Determine GPU count from TP size
    tp_size = config.get("trainer", {}).get("tensor_model_parallel_size", 1)
    num_gpus = max(1, tp_size)

    print("Running 1-step verification")
    print(f"  Config: {config_path}")
    print(f"  TP size: {tp_size}, GPUs: {num_gpus}")
    print()

    # Write temp config and launch
    temp_path = write_temp_config(patched, original_config_path=config_path)
    print("Starting training...")

    try:
        result = launch_training(str(temp_path), num_gpus=num_gpus, timeout=300)

        stdout = result.stdout
        stderr = result.stderr

        if result.returncode != 0:
            print("Training failed!")
            if stderr:
                print("STDERR (last 50 lines):")
                for line in stderr.strip().split("\n")[-50:]:
                    print(f"  {line}")
            sys.exit(1)

    except Exception as e:
        print(f"Error launching training: {e}")
        sys.exit(1)

    # Parse results — training logs go to stderr via Python logging
    metrics = parse_metrics_from_stdout(stderr)
    if not metrics.get("loss"):
        metrics = parse_metrics_from_stdout(stdout)

    # Print results
    print("\n" + "=" * 50)
    print("Step 1 Results")
    print("=" * 50)

    if metrics.get("loss") is not None:
        print(f"  Loss:       {metrics['loss']:.6f}")
    else:
        print("  Loss:       [not found in output]")

    if args.verbose:
        if metrics.get("grad_norm") is not None:
            print(f"  Grad Norm:  {metrics['grad_norm']:.6f}")
        if metrics.get("param_norm") is not None:
            print(f"  Param Norm: {metrics['param_norm']:.6f}")
        if metrics.get("iter_time") is not None:
            print(f"  Step Time:  {metrics['iter_time']:.4f}s")
        if metrics.get("tokens_per_second") is not None:
            print(f"  Throughput: {metrics['tokens_per_second']:.0f} tokens/s")
        if metrics.get("tflops_per_gpu") is not None:
            print(f"  TFLOPS/GPU: {metrics['tflops_per_gpu']:.2f}")
    else:
        if metrics.get("grad_norm") is not None:
            print(f"  Grad Norm:  {metrics['grad_norm']:.6f}")

    # Reference loss comparison
    if args.reference_loss is not None:
        print()
        if metrics.get("loss") is not None:
            diff = abs(metrics["loss"] - args.reference_loss)
            passed = diff <= args.tolerance
            print(f"  Reference Loss: {args.reference_loss:.6f}")
            print(f"  Difference:     {diff:.6f}")
            print(f"  Tolerance:      {args.tolerance}")
            print(f"  Status:         {'PASS' if passed else 'FAIL'}")
        else:
            print("  Cannot compare — loss not found in output.")

    # Write output
    if args.output:
        output = {
            "config": str(config_path),
            "tp_size": tp_size,
            "metrics": {k: v for k, v in metrics.items() if v is not None},
        }
        if args.reference_loss is not None and metrics.get("loss") is not None:
            output["reference_loss"] = args.reference_loss
            output["difference"] = abs(metrics["loss"] - args.reference_loss)
            output["passed"] = abs(metrics["loss"] - args.reference_loss) <= args.tolerance

        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(output, f, indent=2)
        print(f"\nResults written to: {output_path}")

    print()
