#!/usr/bin/env python3
"""
Memory Profiler for Muon vs AdamW Optimizer Comparison

This script runs short training tests with different micro-batch sizes
and reports GPU memory usage for both optimizers.

Expected result: Muon should use ~50% less optimizer state memory
than AdamW because it only needs one momentum buffer vs two for AdamW.
"""

import os
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


@dataclass
class MemoryStats:
    """GPU memory statistics."""

    allocated_mb: float
    reserved_mb: float
    peak_allocated_mb: float
    peak_reserved_mb: float


@dataclass
class TestResult:
    """Result of a single test run."""

    optimizer: str
    micro_batch_size: int
    gradient_accumulation_steps: int
    success: bool
    memory_stats: MemoryStats | None = None
    error_message: str = ""


def create_test_config(
    optimizer: str,
    micro_batch_size: int,
    gradient_accumulation_steps: int,
    steps: int,
    output_path: str,
) -> str:
    """Create a test configuration file."""

    config = f"""# Memory test config for {optimizer}
trainer:
  micro_batch_size: {micro_batch_size}
  train_batch_size: 128
  gradient_accumulation_steps: {gradient_accumulation_steps}
  tensor_model_parallel_size: 1
  save_checkpoint_steps: 10000
  log_interval: 1
  model_path: /tmp/mem_test_{optimizer}
  use_flash_attn: true
  vocab_padding_unit: 128

operation:
  train_steps: {steps}
  eval_interval: 10000
  eval_samples: 100
  activation_recompute: false
  no_save: true

model:
  num_attention_heads: 16
  num_attention_groups: 16
  head_dim: 64
  max_seq_len: 1024
  max_position_embeddings: 1024
  num_layers: 24
  d_model: 1024
  d_ffn: 4096
  untie_embed: false
  reset_position_ids: false
  reset_attention_mask: false
  eod_mask_loss: false
  precision: bf16
  vocab_name_or_path: gpt2
  tokenizer_type: bbpe
  hf_model_type: gpt2
  hf_architecture: GPT2LMHeadModel

data:
  config_path: configs/data/openwebtext_test.yaml
  task_type: pretrain

optim:
  optimizer: {optimizer}
  lr_scheduler: cosine
  max_lr: 0.00042
  min_lr: 0.000042
  warmup_steps: 500
  annealing_steps: 2000
  weight_decay: 0.1
  muon_momentum: 0.95
  muon_newton_schulz_steps: 5
  adam_beta1: 0.9
  adam_beta2: 0.95
  adam_eps: 1.0e-8
  clip_grad: 1.0

init:
  seed: 42
  init_std: 0.02

parallel:
  use_distributed_optimizer: true

utils:
  tensorboard_dir: /tmp/tensorboard/mem_test_{optimizer}
"""

    with open(output_path, "w") as f:
        f.write(config)

    return output_path


def parse_memory_from_output(output: str) -> MemoryStats | None:
    """Parse memory statistics from training output."""
    # Look for memory stats in output
    allocated = reserved = peak_allocated = peak_reserved = None

    # Try various patterns
    patterns = {
        "allocated": [
            r"Allocated[:\s]+([\d.]+)\s*(MB|MiB|GB|GiB)",
            r"memory allocated[:\s]+([\d.]+)\s*(MB|MiB|GB|GiB)",
        ],
        "reserved": [
            r"Reserved[:\s]+([\d.]+)\s*(MB|MiB|GB|GiB)",
            r"memory reserved[:\s]+([\d.]+)\s*(MB|MiB|GB|GiB)",
        ],
        "peak_allocated": [
            r"Peak allocated[:\s]+([\d.]+)\s*(MB|MiB|GB|GiB)",
            r"peak memory[:\s]+([\d.]+)\s*(MB|MiB|GB|GiB)",
        ],
        "peak_reserved": [
            r"Peak reserved[:\s]+([\d.]+)\s*(MB|MiB|GB|GiB)",
        ],
    }

    def to_mb(value: float, unit: str) -> float:
        unit = unit.upper()
        if unit in ("GB", "GIB"):
            return value * 1024
        return value

    for line in output.split("\n"):
        for name, pattern_list in patterns.items():
            for pattern in pattern_list:
                match = re.search(pattern, line, re.IGNORECASE)
                if match:
                    value = float(match.group(1))
                    unit = match.group(2)
                    mb = to_mb(value, unit)
                    if name == "allocated":
                        allocated = mb
                    elif name == "reserved":
                        reserved = mb
                    elif name == "peak_allocated":
                        peak_allocated = mb
                    elif name == "peak_reserved":
                        peak_reserved = mb

    if any([allocated, reserved, peak_allocated, peak_reserved]):
        return MemoryStats(
            allocated_mb=allocated or 0,
            reserved_mb=reserved or 0,
            peak_allocated_mb=peak_allocated or allocated or 0,
            peak_reserved_mb=peak_reserved or reserved or 0,
        )
    return None


def run_test(
    optimizer: str, micro_batch_size: int, gradient_accumulation_steps: int, steps: int = 3
) -> TestResult:
    """Run a single training test and capture memory usage."""

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        config_path = f.name
        create_test_config(
            optimizer=optimizer,
            micro_batch_size=micro_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            steps=steps,
            output_path=config_path,
        )

    try:
        result = subprocess.run(
            ["torchrun", "--nproc_per_node=2", "-m", "ironcore", "train", "--config", config_path],
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            timeout=300,  # 5 minute timeout
            check=False,
        )

        if result.returncode != 0 and not result.stdout:
            # Command failed before producing output (e.g., environment setup error)
            return TestResult(
                optimizer=optimizer,
                micro_batch_size=micro_batch_size,
                gradient_accumulation_steps=gradient_accumulation_steps,
                success=False,
                error_message=f"Trial failed to start: {result.stderr.strip() or 'Unknown error'}",
            )

        output = result.stdout + result.stderr
        memory_stats = parse_memory_from_output(output)

        success = result.returncode == 0 and "Step 3" in output

        return TestResult(
            optimizer=optimizer,
            micro_batch_size=micro_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            success=success,
            memory_stats=memory_stats,
            error_message="" if success else result.stderr[-500:] if result.stderr else "",
        )

    except subprocess.TimeoutExpired:
        return TestResult(
            optimizer=optimizer,
            micro_batch_size=micro_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            success=False,
            error_message="Test timed out",
        )
    except Exception as e:
        return TestResult(
            optimizer=optimizer,
            micro_batch_size=micro_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            success=False,
            error_message=str(e),
        )
    finally:
        os.unlink(config_path)


def main():
    """Run memory profiling tests."""
    print("=" * 70)
    print("MEMORY PROFILER: Muon vs AdamW Optimizer Comparison")
    print("=" * 70)
    print()
    print("Theory: Muon should use ~50% less optimizer state memory")
    print("  - Muon: 1 momentum buffer per parameter")
    print("  - AdamW: 2 momentum buffers (exp_avg, exp_avg_sq)")
    print()

    # Test configurations
    micro_batch_sizes = [2, 4, 8]
    steps = 3
    num_gpus = 2

    results = []

    for mbs in micro_batch_sizes:
        # Calculate gradient accumulation to maintain batch size 128
        gas = max(1, 128 // mbs // num_gpus)

        print(f"\n{'=' * 70}")
        print(f"Testing: micro_batch_size={mbs}, gradient_accumulation_steps={gas}")
        print("=" * 70)

        for optimizer in ["muon", "adam"]:
            print(f"\n  [{optimizer.upper()}] Running {steps} steps...", end=" ", flush=True)

            result = run_test(
                optimizer=optimizer,
                micro_batch_size=mbs,
                gradient_accumulation_steps=gas,
                steps=steps,
            )
            results.append(result)

            if result.success:
                print("SUCCESS")
                if result.memory_stats:
                    ms = result.memory_stats
                    print(f"    Peak Allocated: {ms.peak_allocated_mb:.1f} MB")
                    print(f"    Peak Reserved:  {ms.peak_reserved_mb:.1f} MB")
            else:
                print(f"FAILED - {result.error_message[:100]}")

    # Summary table
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print()
    print(f"{'MBS':>4} | {'Optimizer':>8} | {'GAS':>4} | {'Status':>8} | {'Peak Mem (MB)':>14}")
    print("-" * 70)

    for r in results:
        status = "OK" if r.success else "FAIL"
        mem = f"{r.memory_stats.peak_allocated_mb:.1f}" if r.memory_stats else "N/A"
        print(
            f"{r.micro_batch_size:>4} | {r.optimizer:>8} | {r.gradient_accumulation_steps:>4} | {status:>8} | {mem:>14}"
        )

    # Calculate memory savings if both optimizers succeeded for same MBS
    print("\n" + "-" * 70)
    print("Memory Savings Analysis")
    print("-" * 70)

    for mbs in micro_batch_sizes:
        muon_result = next(
            (
                r
                for r in results
                if r.optimizer == "muon" and r.micro_batch_size == mbs and r.success
            ),
            None,
        )
        adamw_result = next(
            (
                r
                for r in results
                if r.optimizer == "adam" and r.micro_batch_size == mbs and r.success
            ),
            None,
        )

        if muon_result and adamw_result and muon_result.memory_stats and adamw_result.memory_stats:
            muon_mem = muon_result.memory_stats.peak_allocated_mb
            adamw_mem = adamw_result.memory_stats.peak_allocated_mb
            savings = (1 - muon_mem / adamw_mem) * 100 if adamw_mem > 0 else 0
            print(
                f"  MBS={mbs}: Muon={muon_mem:.1f}MB, AdamW={adamw_mem:.1f}MB, Savings={savings:.1f}%"
            )
        else:
            print(f"  MBS={mbs}: Incomplete data for comparison")


if __name__ == "__main__":
    main()
