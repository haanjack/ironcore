# Copyright (c) 2025-2026 Jaegeun Han
#
# SPDX-License-Identifier: Apache-2.0
"""Scaling analysis — run training at multiple scales and fit scaling laws."""

import json
import sys
from argparse import Namespace
from pathlib import Path
from typing import Any

from .utils import (
    deep_merge,
    launch_training,
    load_yaml_config,
    parse_losses_from_stdout,
    print_results_table,
    write_temp_config,
)


def run_analyze_scaling(args: Namespace) -> None:
    """Run training at multiple scales, collect losses, and fit scaling laws.

    Args:
        args: Command-line arguments.
            - config: Base training config YAML
            - scale_dimension: What to scale (model, batch, compute)
            - model_sizes: Comma-separated model config names
            - batch_sizes: Comma-separated batch sizes
            - num_steps: Steps per scale point
            - output_dir: Output directory
            - fit_law: Fit Chinchilla-style scaling law
            - plot: Generate scaling law plots
    """
    config_path = Path(args.config)

    if not config_path.exists():
        print(f"Error: Configuration file not found: {config_path}")
        sys.exit(1)

    config = load_yaml_config(config_path)
    base_tp = config.get("trainer", {}).get("tensor_model_parallel_size", 1)
    # Place temp configs in same directory as original for relative reference resolution
    config_dir = config_path.resolve().parent
    temp_files: list[Path] = []

    try:
        # Build scale points based on dimension
        if args.scale_dimension == "model":
            if not args.model_sizes:
                print("Error: --model-sizes required for model scaling")
                print("  Example: --model-sizes gpt2-micro,gpt2-tiny,gpt2-small-test,gpt2-small")
                sys.exit(1)

            scale_points = _build_model_scale_points(args.model_sizes, config, args.num_steps)
        elif args.scale_dimension == "batch":
            if not args.batch_sizes:
                print("Error: --batch-sizes required for batch scaling")
                print("  Example: --batch-sizes 32,64,128,256")
                sys.exit(1)

            scale_points = _build_batch_scale_points(
                [int(b) for b in args.batch_sizes.split(",")],
                config,
                args.num_steps,
            )
        else:
            print(f"Error: Unsupported scale dimension: {args.scale_dimension}")
            sys.exit(1)

        print("Scaling Analysis")
        print(f"  Dimension: {args.scale_dimension}")
        print(f"  Config: {config_path}")
        print(f"  Steps per point: {args.num_steps}")
        print(f"  Scale points: {len(scale_points)}")
        print()

        # Run each scale point
        results = []
        for i, point in enumerate(scale_points):
            label = point["label"]
            overrides = point["overrides"]
            num_gpus = point.get("num_gpus", base_tp)
            scale_value = point.get("scale_value", 0)

            print(f"[{i + 1}/{len(scale_points)}] Running: {label}...")

            patched = deep_merge(config, overrides)
            variant_path = config_dir / f".ironcore_scaling_{id(config) % 100000}_{i}.yaml"
            temp_files.append(variant_path)
            write_temp_config(patched, output_path=variant_path, original_config_path=config_path)

            try:
                proc = launch_training(str(variant_path), num_gpus=num_gpus, timeout=3600)

                if proc.returncode != 0:
                    print(f"  FAILED (exit code {proc.returncode})")
                    results.append(
                        {
                            "label": label,
                            "scale_value": scale_value,
                            "status": "FAILED",
                            "final_loss": None,
                        }
                    )
                    continue

                losses = parse_losses_from_stdout(proc.stderr or proc.stdout)
                final_loss = losses[-1] if losses else None
                results.append(
                    {
                        "label": label,
                        "scale_value": scale_value,
                        "status": "OK",
                        "final_loss": final_loss,
                        "num_steps": len(losses),
                        "losses": losses,
                    }
                )
                if final_loss is not None:
                    print(f"  Final loss: {final_loss:.6f}")

            except Exception as e:
                print(f"  ERROR: {e}")
                results.append(
                    {
                        "label": label,
                        "scale_value": scale_value,
                        "status": "ERROR",
                        "error": str(e),
                        "final_loss": None,
                    }
                )

        print()

        # Print results table
        print_results_table(
            results,
            ["label", "scale_value", "final_loss", "status"],
            "Scaling Results",
        )
        print()

        # Fit scaling law
        successful = [
            {**r, "log_scale": float(r["scale_value"]) if r["scale_value"] else 0}
            for r in results
            if r["status"] == "OK" and r["final_loss"] is not None and r["scale_value"]
        ]

        fit_results: dict[str, Any] = {}
        if args.fit_law and len(successful) >= 3:
            fit_results = _fit_scaling_law(successful)
            if fit_results:
                print("Scaling Law Fit (power law: L(N) = a * N^b + c):")
                print(f"  a = {fit_results.get('a', '?')}")
                print(f"  b = {fit_results.get('b', '?')}")
                print(f"  c = {fit_results.get('c', '?')}")
                print(f"  R² = {fit_results.get('r_squared', '?')}")
                print()

        # Generate plot
        if args.plot and successful:
            _generate_scaling_plot(successful, fit_results, args.output_dir)

        # Save results
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        results_path = output_dir / "scaling_results.json"
        with open(results_path, "w") as f:
            json.dump(
                {
                    "dimension": args.scale_dimension,
                    "config": str(config_path),
                    "num_steps": args.num_steps,
                    "results": results,
                    "fit": fit_results,
                },
                f,
                indent=2,
            )
        print(f"Results saved to: {results_path}")

    finally:
        for f in temp_files:
            f.unlink(missing_ok=True)

    print()


def _build_model_scale_points(model_names: str, config: dict, num_steps: int) -> list[dict]:
    """Build scale points for model-size scaling.

    Args:
        model_names: Comma-separated model config names.
        config: Base config dict.
        num_steps: Training steps per point.

    Returns:
        List of scale point dicts.
    """
    points = []
    for name in model_names.split(","):
        name = name.strip()
        # Load model config to estimate params
        model_config_path = Path("configs/model") / f"{name}.yaml"
        if not model_config_path.exists():
            print(f"  Warning: Model config not found: {model_config_path}, skipping.")
            continue

        model_config = load_yaml_config(model_config_path)
        params = _estimate_params_from_config(model_config)

        points.append(
            {
                "label": name,
                "scale_value": params,
                "overrides": {
                    "model": name,
                    "trainer": {"log_interval": 1},
                    "operation": {"train_steps": num_steps, "no_save": True},
                    "profiler": {
                        "gpu_profiler": False,
                        "torch_profiler": False,
                        "comm_profiler": False,
                        "layer_timing": False,
                    },
                },
                "num_gpus": 1,
            }
        )

    return points


def _build_batch_scale_points(
    batch_sizes: list[int], base_config: dict, num_steps: int
) -> list[dict]:
    """Build scale points for batch-size scaling.

    Args:
        batch_sizes: List of batch sizes to try.
        config: Base config dict.
        num_steps: Training steps per point.

    Returns:
        List of scale point dicts.
    """
    base_micro_batch = base_config.get("trainer", {}).get("micro_batch_size", 4)
    base_tp = base_config.get("trainer", {}).get("tensor_model_parallel_size", 1)
    seq_len = base_config.get("data", {}).get("seq_length", 1024)

    points = []
    for batch in batch_sizes:
        # Ensure gradient accumulation is consistent
        grad_accum = max(1, batch // (base_micro_batch * max(1, 1)))  # DP=1 assumed
        actual_batch = base_micro_batch * grad_accum

        points.append(
            {
                "label": f"batch={actual_batch}",
                "scale_value": actual_batch * seq_len,  # tokens per step
                "overrides": {
                    "trainer": {
                        "train_batch_size": actual_batch,
                        "gradient_accumulation_steps": grad_accum,
                        "log_interval": 1,
                    },
                    "operation": {"train_steps": num_steps, "no_save": True},
                    "profiler": {
                        "gpu_profiler": False,
                        "torch_profiler": False,
                        "comm_profiler": False,
                        "layer_timing": False,
                    },
                },
                "num_gpus": base_tp,
            }
        )

    return points


def _estimate_params_from_config(model_config: dict) -> int:
    """Estimate parameter count from model config dict.

    Args:
        model_config: Model config as a dict.

    Returns:
        Estimated parameter count.
    """
    layers = model_config.get("num_layers", 0)
    d_model = model_config.get("d_model", 0)
    d_ffn = model_config.get("d_ffn", 0)
    heads = model_config.get("num_attention_heads", 0)
    head_dim = model_config.get("head_dim", 64)
    groups = model_config.get("num_attention_groups", heads)
    vocab_size = model_config.get("vocab_size", 50257)

    embed = vocab_size * d_model
    attn = d_model * (heads * head_dim + 2 * groups * head_dim) + (heads * head_dim) * d_model
    mlp = 2 * d_model * d_ffn
    ln = 4 * d_model
    return embed + layers * (attn + mlp + ln) + 2 * d_model


def _fit_scaling_law(data: list[dict]) -> dict[str, Any]:
    """Fit power law L(N) = a * N^b + c to scaling data.

    Args:
        data: List of dicts with 'scale_value' and 'final_loss' keys.

    Returns:
        Dict with fit parameters a, b, c, and r_squared.
    """
    try:
        import numpy as np
        from scipy.optimize import curve_fit

        n_values = np.array([d["scale_value"] for d in data], dtype=np.float64)
        losses = np.array([d["final_loss"] for d in data], dtype=np.float64)

        def power_law(n, a, b, c):
            return a * np.power(n, b) + c

        # Initial guesses
        p0 = [10.0, -0.1, losses.min()]

        popt, _ = curve_fit(power_law, n_values, losses, p0=p0, maxfev=10000)
        a, b, c = popt

        # Compute R²
        predicted = power_law(n_values, a, b, c)
        ss_res = np.sum((losses - predicted) ** 2)
        ss_tot = np.sum((losses - np.mean(losses)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

        return {
            "a": float(a),
            "b": float(b),
            "c": float(c),
            "r_squared": float(r_squared),
        }

    except ImportError:
        print("  Warning: scipy not installed, skipping scaling law fit")
        print("  Install with: pip install scipy")
        return {}
    except Exception as e:
        print(f"  Warning: Scaling law fit failed: {e}")
        return {}


def _generate_scaling_plot(data: list[dict], fit: dict, output_dir: str) -> None:
    """Generate a scaling law plot.

    Args:
        data: List of scaling results.
        fit: Fit parameters dict.
        output_dir: Output directory for the plot.
    """
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np

        n_values = [d["scale_value"] for d in data]
        losses = [d["final_loss"] for d in data]

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.scatter(n_values, losses, color="blue", s=80, zorder=5, label="Measured")

        if fit and "a" in fit:
            n_range = np.linspace(min(n_values), max(n_values), 100)
            fitted = fit["a"] * np.power(n_range, fit["b"]) + fit["c"]
            ax.plot(n_range, fitted, "r--", label=f"Fit (R²={fit.get('r_squared', 0):.3f})")

        ax.set_xlabel("Scale (parameters or tokens)")
        ax.set_ylabel("Loss")
        ax.set_title("Scaling Analysis")
        ax.legend()
        ax.grid(True, alpha=0.3)

        output_path = Path(output_dir) / "scaling_plot.png"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(output_path), dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Plot saved to: {output_path}")

    except ImportError:
        print("  Warning: matplotlib not installed, skipping plot")
        print("  Install with: pip install matplotlib")
    except Exception as e:
        print(f"  Warning: Plot generation failed: {e}")
