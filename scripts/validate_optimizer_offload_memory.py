"""Optimizer offload validation: loss parity + VRAM measurement.

Compares baseline (no offload) vs optimizer_offload only across two model sizes.
Reports: loss trajectory, final loss relative error, peak VRAM, VRAM savings.

Usage:
    python scripts/validate_m1_memory.py
"""

import os
import sys

import torch
import torch.nn.functional as F

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tests.fixtures.config_fixtures import create_test_config
from tests.integration.offload.conftest import (
    create_mock_data_iterator,
    create_mock_evaluators,
)

from ironcore.global_vars import reset_global_states
from ironcore.trainers import LanguageModelTrainer

NUM_STEPS = 50
BATCH_SIZE = 2
SEQ_LEN = 256


def _make_config(d_model=768, d_ffn=3072, num_layers=4, num_heads=12, **offload_overrides):
    from ironcore.config import OffloadConfig

    offload = OffloadConfig()
    for k, v in offload_overrides.items():
        setattr(offload, k, v)

    config = create_test_config(
        d_model=d_model,
        d_ffn=d_ffn,
        num_layers=num_layers,
        num_attention_heads=num_heads,
        num_attention_groups=num_heads,
        head_dim=64,
        max_seq_len=SEQ_LEN,
        dropout_attn=0.0,
        dropout_mlp=0.0,
        dropout_embd=0.0,
        precision="bfloat16",
        seed=42,
    )
    config.operation.train_steps = NUM_STEPS + 10
    config.trainer.micro_batch_size = BATCH_SIZE
    config.trainer.train_batch_size = BATCH_SIZE
    config.trainer.gradient_accumulation_steps = 1
    config.parallel.world_size = 1
    config.offload = offload
    return config


def _create_forward_step_func():
    step_counter = [0]

    def forward_step(model, _data_iterator):
        device = next(model.parameters()).device
        torch.manual_seed(42 + step_counter[0])
        step_counter[0] += 1
        input_ids = torch.randint(0, 1000, (BATCH_SIZE, SEQ_LEN), device=device)
        labels = input_ids.clone()
        logits = model(input_ids, labels=None)
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()
        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
        )
        return loss

    return forward_step


def _run_training_with_memory(config, num_steps, label=""):
    """Run N training steps, track loss and peak VRAM at each step."""
    reset_global_states()
    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", "29500")
    os.environ.setdefault("LOCAL_RANK", "0")
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    if not torch.distributed.is_initialized():
        torch.distributed.init_process_group(backend="nccl", rank=0, world_size=1)

    from unittest.mock import patch

    losses = []
    peak_vram_mb = []
    allocated_vram_mb = []

    with (
        patch(
            "ironcore.trainers.base_trainer.get_data_iterator",
            return_value=create_mock_data_iterator(),
        ),
        patch(
            "ironcore.trainers.base_trainer.get_evaluators",
            return_value=create_mock_evaluators(),
        ),
    ):
        trainer = LanguageModelTrainer(
            config,
            _create_forward_step_func(),
            F.cross_entropy,
        )
        trainer._initialize()

        for step in range(num_steps):
            torch.cuda.reset_peak_memory_stats()

            loss, _, _ = trainer.train_step(step=step)
            losses.append(loss)

            peak = torch.cuda.max_memory_allocated()
            allocated = torch.cuda.memory_allocated()
            peak_vram_mb.append(peak / 1024**2)
            allocated_vram_mb.append(allocated / 1024**2)

        trainer._finalize_process()

    return losses, peak_vram_mb, allocated_vram_mb


def _print_section(title):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")


def main():
    if not torch.cuda.is_available():
        print("CUDA required for optimizer_offload validation")
        sys.exit(1)

    configs = [
        {
            "label": "GPT-small (4L, 768d, ~67M params)",
            "d_model": 768, "d_ffn": 3072, "num_layers": 4, "num_heads": 12,
        },
        {
            "label": "GPT-medium (8L, 1024d, ~246M params)",
            "d_model": 1024, "d_ffn": 4096, "num_layers": 8, "num_heads": 16,
        },
    ]

    print()
    print("=" * 70)
    print("  OPTIMIZER OFFLOAD VALIDATION REPORT")
    print("  CPU-compute path: AdamW math on CPU, grad+delta transfers only")
    print("=" * 70)

    for cfg in configs:
        label = cfg.pop("label")
        _print_section(label)

        # Baseline
        config_base = _make_config(**cfg)
        losses_base, peak_base, alloc_base = _run_training_with_memory(
            config_base, NUM_STEPS, "baseline"
        )

        # optimizer_offload only (CPU-compute path)
        config_m1 = _make_config(optimizer_offload=True, enabled=True, **cfg)
        losses_m1, peak_m1, alloc_m1 = _run_training_with_memory(
            config_m1, NUM_STEPS, "optimizer_offload"
        )

        # Loss report
        print(f"\n  Loss trajectory (first 5 + last 5 of {NUM_STEPS} steps):")
        print(f"  {'Step':<6} {'Baseline':>10} {'optimizer_offload':>18} {'Diff':>10}")
        print(f"  {'-'*40}")
        indices = list(range(5)) + list(range(NUM_STEPS - 5, NUM_STEPS))
        for i in indices:
            if i == 5:
                print(f"  {'...':<6}")
            diff = losses_m1[i] - losses_base[i]
            print(f"  {i:<6} {losses_base[i]:>10.4f} {losses_m1[i]:>10.4f} {diff:>+10.4f}")

        rel_err = abs(losses_base[-1] - losses_m1[-1]) / (abs(losses_base[-1]) + 1e-8)
        print(f"\n  Final loss: baseline={losses_base[-1]:.4f}  optimizer_offload={losses_m1[-1]:.4f}  rel_err={rel_err:.4f}")
        verdict = "PASS" if rel_err < 0.01 else "FAIL"
        print(f"  Loss parity: {verdict} (threshold: 1%)")

        # VRAM report
        # Use steady-state peak (skip step 0 which includes model init)
        steady_peak_base = max(peak_base[1:])
        steady_peak_m1 = max(peak_m1[1:])
        steady_alloc_base = max(alloc_base[1:])
        steady_alloc_m1 = max(alloc_m1[1:])

        vram_delta = steady_peak_base - steady_peak_m1
        vram_pct = (vram_delta / steady_peak_base) * 100 if steady_peak_base > 0 else 0

        print(f"\n  VRAM (steady-state peak, steps 1-{NUM_STEPS-1}):")
        print(f"  {'Metric':<25} {'Baseline':>10} {'optimizer_offload':>18} {'Delta':>10}")
        print(f"  {'-'*55}")
        print(f"  {'Peak VRAM (MB)':<25} {steady_peak_base:>10.1f} {steady_peak_m1:>10.1f} {vram_delta:>+10.1f}")
        print(f"  {'Allocated VRAM (MB)':<25} {steady_alloc_base:>10.1f} {steady_alloc_m1:>10.1f} {steady_alloc_base - steady_alloc_m1:>+10.1f}")

        vram_verdict = "PASS" if vram_delta > 0 else "NEUTRAL"
        print(f"\n  VRAM savings: {vram_pct:+.1f}% ({vram_delta:+.1f} MB) — {vram_verdict}")

        # Per-step peak VRAM comparison
        print("\n  Per-step peak VRAM (selected steps):")
        print(f"  {'Step':<6} {'Baseline':>10} {'optimizer_offload':>18} {'Delta':>10}")
        print(f"  {'-'*40}")
        for i in [1, 5, 10, 20, 30, 40, 49]:
            if i < NUM_STEPS:
                d = peak_base[i] - peak_m1[i]
                print(f"  {i:<6} {peak_base[i]:>10.1f} {peak_m1[i]:>10.1f} {d:>+10.1f}")

        cfg["label"] = label  # restore for next iteration if needed

    print(f"\n{'='*70}")
    print("  SUMMARY")
    print(f"{'='*70}")
    print("  Optimizer offload now uses CPU-compute path when params are on GPU.")
    print("  AdamW math runs on CPU (AVX-512/SIMD via MKL). Only grad (GPU->CPU)")
    print("  and delta (CPU->GPU) are transferred. Optimizer states never leave CPU.")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
