# Copyright (c) 2025-2026 Jaegeun Han
#
# SPDX-License-Identifier: Apache-2.0
"""Parallelism correctness verification — compare training across TP/DP/FSDP configs."""

import json
import sys
from argparse import Namespace
from pathlib import Path

from .utils import (
    deep_merge,
    launch_training,
    load_yaml_config,
    parse_losses_from_stdout,
    print_results_table,
    write_temp_config,
)


def run_verify_parity(args: Namespace) -> None:
    """Verify parallelism correctness by comparing loss curves across configurations.

    Args:
        args: Command-line arguments.
            - config: Base training config YAML
            - mode: Parity mode (tp, dp, fsdp)
            - tp_sizes: Comma-separated TP sizes (for tp mode)
            - num_steps: Steps to run per config
            - tolerance: Loss tolerance for matching
            - output: Optional output file for results JSON
            - seed: Random seed for reproducibility
    """
    config_path = Path(args.config)

    if not config_path.exists():
        print(f"Error: Configuration file not found: {config_path}")
        sys.exit(1)

    config = load_yaml_config(config_path)
    base_tp = config.get("trainer", {}).get("tensor_model_parallel_size", 1)

    # Build variant configurations based on mode
    variants: list[dict] = []

    if args.mode == "tp":
        tp_sizes = [int(s.strip()) for s in args.tp_sizes.split(",")]
        for tp in tp_sizes:
            variants.append({
                "label": f"TP={tp}",
                "overrides": {
                    "trainer": {"tensor_model_parallel_size": tp, "log_interval": 1},
                    "operation": {"train_steps": args.num_steps, "no_save": True},
                    "init": {"seed": args.seed},
                    "profiler": {
                        "gpu_profiler": False, "torch_profiler": False,
                        "comm_profiler": False, "layer_timing": False,
                    },
                },
                "num_gpus": max(1, tp),
            })
    elif args.mode == "dp":
        variants.extend([
            {
                "label": f"TP={base_tp}, DP=1",
                "overrides": {
                    "trainer": {"log_interval": 1},
                    "operation": {"train_steps": args.num_steps, "no_save": True},
                    "init": {"seed": args.seed},
                    "profiler": {
                        "gpu_profiler": False, "torch_profiler": False,
                        "comm_profiler": False, "layer_timing": False,
                    },
                },
                "num_gpus": base_tp,
            },
            {
                "label": f"TP={base_tp}, DP=2",
                "overrides": {
                    "trainer": {"log_interval": 1},
                    "operation": {"train_steps": args.num_steps, "no_save": True},
                    "init": {"seed": args.seed},
                    "profiler": {
                        "gpu_profiler": False, "torch_profiler": False,
                        "comm_profiler": False, "layer_timing": False,
                    },
                },
                "num_gpus": base_tp * 2,
            },
        ])
    elif args.mode == "fsdp":
        variants.extend([
            {
                "label": "FSDP=off",
                "overrides": {
                    "trainer": {"log_interval": 1},
                    "parallel": {"use_fsdp": False},
                    "operation": {"train_steps": args.num_steps, "no_save": True},
                    "init": {"seed": args.seed},
                    "profiler": {
                        "gpu_profiler": False, "torch_profiler": False,
                        "comm_profiler": False, "layer_timing": False,
                    },
                },
                "num_gpus": base_tp,
            },
            {
                "label": "FSDP=on",
                "overrides": {
                    "trainer": {"log_interval": 1},
                    "parallel": {"use_fsdp": True},
                    "operation": {"train_steps": args.num_steps, "no_save": True},
                    "init": {"seed": args.seed},
                    "profiler": {
                        "gpu_profiler": False, "torch_profiler": False,
                        "comm_profiler": False, "layer_timing": False,
                    },
                },
                "num_gpus": base_tp,
            },
        ])

    if len(variants) < 2:
        print("Error: Need at least 2 variants to compare.")
        sys.exit(1)

    print(f"Parallelism Parity Verification")
    print(f"  Mode: {args.mode}")
    print(f"  Config: {config_path}")
    print(f"  Steps: {args.num_steps}")
    print(f"  Tolerance: {args.tolerance}")
    print(f"  Seed: {args.seed}")
    print(f"  Variants: {len(variants)}")
    print()

    # Write temp configs in the same directory as the original config
    # so relative references (model: gpt2-micro) resolve correctly.
    config_dir = config_path.resolve().parent
    temp_files: list[Path] = []

    try:
        # Run each variant
        results = []
        for variant in variants:
            label = variant["label"]
            overrides = variant["overrides"]
            num_gpus = variant["num_gpus"]

            print(f"Running: {label} ({num_gpus} GPU(s))...")

            patched = deep_merge(config, overrides)
            safe_label = label.replace("=", "_").replace(",", "_").replace(" ", "_")
            variant_path = config_dir / f".ironcore_parity_{id(config) % 100000}_{safe_label}.yaml"
            temp_files.append(variant_path)
            write_temp_config(patched, output_path=variant_path, original_config_path=config_path)

            try:
                proc = launch_training(str(variant_path), num_gpus=num_gpus, timeout=600)

                if proc.returncode != 0:
                    print(f"  FAILED (exit code {proc.returncode})")
                    if proc.stderr:
                        for line in proc.stderr.strip().split("\n")[-10:]:
                            print(f"    {line}")
                    results.append({
                        "label": label,
                        "status": "FAILED",
                        "losses": [],
                        "num_gpus": num_gpus,
                    })
                    continue

                losses = parse_losses_from_stdout(proc.stderr or proc.stdout)
                results.append({
                    "label": label,
                    "status": "OK",
                    "losses": losses,
                    "final_loss": losses[-1] if losses else None,
                    "num_losses": len(losses),
                    "num_gpus": num_gpus,
                })
                if losses:
                    print(f"  Final loss: {losses[-1]:.6f} ({len(losses)} steps)")
                else:
                    print(f"  WARNING: No losses found in output")

            except Exception as e:
                print(f"  ERROR: {e}")
                results.append({
                    "label": label,
                    "status": "ERROR",
                    "error": str(e),
                    "losses": [],
                    "num_gpus": num_gpus,
                })

        print()

        # Compare results
        successful = [r for r in results if r["status"] == "OK" and r["losses"]]
        if len(successful) < 2:
            print("Not enough successful runs to compare.")
            print_results_table(results, ["label", "status", "num_losses", "num_gpus"], "Results Summary")
            sys.exit(1)

        # Pairwise comparison between first successful run and all others
        ref = successful[0]
        comparison_rows = []
        all_pass = True

        for run in successful[1:]:
            min_len = min(len(ref["losses"]), len(run["losses"]))
            max_diff = 0.0
            diffs = []

            for i in range(min_len):
                diff = abs(ref["losses"][i] - run["losses"][i])
                diffs.append(diff)
                max_diff = max(max_diff, diff)

            passed = all(d <= args.tolerance for d in diffs) and min_len > 0
            if not passed:
                all_pass = False

            comparison_rows.append({
                "variant_a": ref["label"],
                "variant_b": run["label"],
                "max_diff": f"{max_diff:.2e}",
                "tolerance": f"{args.tolerance:.2e}",
                "steps_compared": min_len,
                "status": "PASS" if passed else "FAIL",
            })

        # Print results
        print_results_table(
            results, ["label", "status", "final_loss", "num_losses", "num_gpus"],
            "Run Summary",
        )
        print()
        print_results_table(
            comparison_rows,
            ["variant_a", "variant_b", "max_diff", "tolerance", "steps_compared", "status"],
            "Parity Comparison",
        )
        print()

        # Overall verdict
        if all_pass:
            print(f"PARITY VERIFIED — all losses within tolerance {args.tolerance}")
        else:
            print(f"PARITY FAILED — max differences exceed tolerance {args.tolerance}")

        # Write output
        if args.output:
            output = {
                "mode": args.mode,
                "config": str(config_path),
                "num_steps": args.num_steps,
                "tolerance": args.tolerance,
                "seed": args.seed,
                "all_pass": all_pass,
                "runs": results,
                "comparisons": comparison_rows,
            }
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w") as f:
                json.dump(output, f, indent=2)
            print(f"\nResults written to: {output_path}")

    finally:
        # Clean up temp config files
        for f in temp_files:
            f.unlink(missing_ok=True)

    print()
