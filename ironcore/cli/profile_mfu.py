# Copyright (c) 2025-2026 Jaegeun Han
#
# SPDX-License-Identifier: Apache-2.0
"""MFU profiling — measure Model FLOP Utilization."""

import json
import sys
from argparse import Namespace
from pathlib import Path

from ironcore.utils import deep_merge, estimate_params, load_yaml_config
from ironcore.utils.subprocess import launch_training, write_temp_config


def register_parser(subparsers) -> None:
    """Register the CLI subcommand arguments."""
    parser = subparsers.add_parser("profile-mfu", help="Profile Model FLOP Utilization")
    parser.add_argument(
        "--config", type=str, required=True, help="Path to training configuration YAML file"
    )
    parser.add_argument("--warmup-steps", type=int, default=3)
    parser.add_argument("--measure-steps", type=int, default=5)
    parser.add_argument(
        "--hardware-peak",
        type=float,
        default=35.6,
        help="Hardware peak TFLOPS/s (default: 35.6 for RTX 3090 bf16)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file for MFU results JSON",
    )
    parser.add_argument(
        "--compare",
        type=str,
        default=None,
        help="Path to previous MFU results JSON for comparison",
    )


def run_profile_mfu(args: Namespace) -> None:
    """Measure Model FLOP Utilization against hardware peak.

    Args:
        args: Command-line arguments.
            - config: Path to training config YAML
            - warmup_steps: Warmup steps (default: 3)
            - measure_steps: Steps to measure (default: 5)
            - hardware_peak: Peak TFLOPS/s (default: 35.6 for RTX 3090)
            - output: Optional output file for results JSON
            - compare: Optional path to previous results for comparison
    """
    config_path = Path(args.config)

    if not config_path.exists():
        print(f"Error: Configuration file not found: {config_path}")
        sys.exit(1)

    config = load_yaml_config(config_path)

    total_steps = args.warmup_steps + args.measure_steps
    tp_size = config.get("trainer", {}).get("tensor_model_parallel_size", 1)
    num_gpus = max(1, tp_size)

    # Extract model info for param estimation
    model_config = config.get("model", {})
    if isinstance(model_config, str):
        # String reference like "gpt2-micro" — load the model config
        model_name = model_config
        model_config_path = Path("configs/model") / f"{model_config}.yaml"
        if model_config_path.exists():
            model_config = load_yaml_config(model_config_path)
        else:
            model_config = {}
    else:
        model_name = model_config.get("hf_model_type", "unknown")

    d_model = model_config.get("d_model", 0)
    d_ffn = model_config.get("d_ffn", 0)
    layers = model_config.get("num_layers", 0)
    heads = model_config.get("num_attention_heads", 0)
    head_dim = model_config.get("head_dim", 64)
    groups = model_config.get("num_attention_groups", heads)
    vocab_size = model_config.get("vocab_size", 50257)
    activation_type = model_config.get("activation_type", "gelu")

    # Estimate parameter count
    num_params = estimate_params(
        d_model, d_ffn, layers, heads, head_dim, groups, vocab_size, activation_type
    )

    # Extract batch/seq info
    trainer_config = config.get("trainer", {})
    global_batch = trainer_config.get("train_batch_size", 0)
    data_config = config.get("data", {})
    seq_len = data_config.get("seq_length", 0)

    # If data config references an external file, load it for seq_length
    if not seq_len and data_config.get("config_path"):
        try:
            ext_data = load_yaml_config(data_config["config_path"])
            seq_len = ext_data.get("seq_length", 1024)
        except Exception:
            seq_len = 1024
    elif not seq_len:
        seq_len = model_config.get("max_seq_len", 1024)

    # Override config for MFU run
    overrides = {
        "operation": {
            "train_steps": total_steps,
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

    print("MFU Profiling")
    print(f"  Model: {model_name} (~{num_params:,} params)")
    print(f"  Batch: {global_batch}, Seq: {seq_len}")
    print(f"  Warmup: {args.warmup_steps} steps, Measure: {args.measure_steps} steps")
    print(f"  GPUs: {num_gpus}, Hardware peak: {args.hardware_peak} TFLOPS/s")
    print()

    # Launch training
    temp_path = write_temp_config(patched, original_config_path=config_path)
    print("Starting training...")

    try:
        result = launch_training(str(temp_path), num_gpus=num_gpus, timeout=1800)
        stdout = result.stdout
        stderr = result.stderr

        if result.returncode != 0:
            print("Training failed!")
            if stderr:
                for line in stderr.strip().split("\n")[-30:]:
                    print(f"  {line}")
            sys.exit(1)

    except Exception as e:
        print(f"Error launching training: {e}")
        sys.exit(1)

    # Parse timing from measure steps — training logs go to stderr
    import re

    combined_output = (stderr or "") + (stdout or "")
    time_pattern = r"iter_time:\s*([\d.]+)s"
    all_times = [float(m) for m in re.findall(time_pattern, combined_output)]

    if len(all_times) < args.measure_steps:
        print(f"Warning: Expected {args.measure_steps} step times, got {len(all_times)}")
        if not all_times:
            print("No timing data found in output. Check training logs.")
            sys.exit(1)

    # Use last measure_steps for average
    measure_times = all_times[-args.measure_steps :]
    avg_step_time = sum(measure_times) / len(measure_times)

    # Compute TFLOPS: FLOPs = 6 * params * tokens, TFLOPS = FLOPs / (time * 1e12)
    tokens_per_step = global_batch * seq_len
    dp_world_size = max(1, num_gpus // tp_size) if tp_size > 0 else 1
    flops = 6 * num_params * tokens_per_step
    achieved_tflops = flops / (avg_step_time * 1e12)
    achieved_tflops_per_gpu = achieved_tflops / max(1, num_gpus)
    mfu_percent = (achieved_tflops_per_gpu / args.hardware_peak) * 100
    throughput = tokens_per_step / avg_step_time

    # Print results
    print("\n" + "=" * 55)
    print("MFU Profile Results")
    print("=" * 55)
    print(f"  Model:          {model_name} (~{num_params:,} params)")
    print(f"  Config:         TP={tp_size}, batch={global_batch}, seq={seq_len}")
    print(f"  Hardware Peak:  {args.hardware_peak} TFLOPS/s")
    print()
    print(f"  Avg step time:  {avg_step_time:.4f}s ({len(measure_times)} steps)")
    print(f"  Tokens/step:    {tokens_per_step:,}")
    print(f"  Throughput:     {throughput:,.0f} tokens/s")
    print()
    print(f"  Achieved:       {achieved_tflops_per_gpu:.2f} TFLOPS/s/GPU")
    print(f"  MFU:            {mfu_percent:.1f}%")

    # Comparison
    if args.compare:
        print()
        compare_path = Path(args.compare)
        if compare_path.exists():
            with open(compare_path) as f:
                prev = json.load(f)
            prev_mfu = prev.get("mfu_percent", 0)
            prev_tflops = prev.get("achieved_tflops_per_gpu", 0)
            delta_mfu = mfu_percent - prev_mfu
            delta_tflops = achieved_tflops_per_gpu - prev_tflops
            print(f"  Comparison ({args.compare}):")
            print(f"    Previous MFU:   {prev_mfu:.1f}%")
            print(f"    Current MFU:    {mfu_percent:.1f}% ({delta_mfu:+.1f}%)")
            print(f"    Previous TFLOPS: {prev_tflops:.2f}")
            print(f"    Current TFLOPS:  {achieved_tflops_per_gpu:.2f} ({delta_tflops:+.2f})")
        else:
            print(f"  Warning: Comparison file not found: {args.compare}")

    # Write output
    results = {
        "model": model_name,
        "num_params": num_params,
        "tp_size": tp_size,
        "dp_world_size": dp_world_size,
        "global_batch_size": global_batch,
        "seq_length": seq_len,
        "tokens_per_step": tokens_per_step,
        "warmup_steps": args.warmup_steps,
        "measure_steps": args.measure_steps,
        "avg_step_time": avg_step_time,
        "step_times": measure_times,
        "achieved_tflops": achieved_tflops,
        "achieved_tflops_per_gpu": achieved_tflops_per_gpu,
        "mfu_percent": round(mfu_percent, 2),
        "throughput_tokens_per_sec": round(throughput, 1),
        "hardware_peak_tflops": args.hardware_peak,
    }

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults written to: {output_path}")

    print()
