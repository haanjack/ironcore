#!/usr/bin/env python3
"""Measure training activation memory vs context length on 13B with full offload."""
import os, sys, time, torch
from pathlib import Path

os.environ.setdefault("MASTER_ADDR", "localhost")
os.environ.setdefault("MASTER_PORT", "29506")
os.environ.setdefault("LOCAL_RANK", "0")
os.environ.setdefault("RANK", "0")
os.environ.setdefault("WORLD_SIZE", "1")

if not torch.distributed.is_initialized():
    torch.distributed.init_process_group(backend="nccl", rank=0, world_size=1)

sys.path.insert(0, str(Path(__file__).parent.parent))

from ironcore.config import *
from ironcore.global_vars import reset_global_states
from ironcore.trainers import LanguageModelTrainer
from unittest.mock import patch
import torch.nn.functional as F
from tests.integration.offload.conftest import create_mock_data_iterator, create_mock_evaluators

SEQ_LENS = [512, 1024, 2048, 4096]

def run_training_steps(seq_len, num_steps=3):
    """Run N training steps at given seq_len, measure VRAM and timing."""
    reset_global_states()

    model_cfg = ModelConfig(
        name="llama-13b",
        d_model=5120,
        d_ffn=13824,
        num_layers=40,
        num_attention_heads=40,
        num_attention_groups=8,
        head_dim=128,
        max_seq_len=seq_len,
        precision="bfloat16",
        ln_type="rmsnorm",
        activation_type="swiglu",
        vocab_name_or_path="gpt2",
    )
    model_cfg.positional_embedding.type = "rope"
    model_cfg.positional_embedding.base = 10000

    offload_cfg = OffloadConfig(
        enabled=True,
        optimizer_offload=True,
        optimizer_state_precision="bf16",
        weight_offload=True,
        weight_prefetch_layers=2,
        weight_storage_precision="bf16",
        activation_spill=True,
        activation_spill_granularity="sub_layer",
        pinned_memory_pool_gb=80.0,
        gpu_staging_pool_mb=0.0,
    )

    main_config = MainConfig(
        model=model_cfg,
        init=InitConfig(seed=42),
        optim=OptimConfig(optimizer="adamw", max_lr=3e-4, min_lr=3e-5, warmup_steps=1, annealing_steps=num_steps + 1, weight_decay=0.1, clip_grad=1.0),
        data=DataConfig(seq_length=seq_len),
        parallel=ParallelConfig(world_size=1, rank=0, local_rank=0),
        trainer=TrainerConfig(micro_batch_size=1, train_batch_size=1, gradient_accumulation_steps=1, model_path="/tmp/validate_ctx", log_interval=1, use_flash_attn=True),
        operation=OperationConfig(train_steps=num_steps + 2, activation_recompute=False, no_save=True),
        utils=UtilsConfig(log_level="WARNING"),
        profiler=ProfilerConfig(),
        peft=PEFTConfig(method="none"),
        alignment=AlignmentConfig(),
        offload=offload_cfg,
    )

    def forward_step(model, _data_iterator):
        device = next(model.parameters()).device
        input_ids = torch.randint(0, 32000, (1, seq_len), device=device)
        labels = input_ids.clone()
        logits = model(input_ids, labels=None)
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()
        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
        )
        return loss

    try:
        with patch('ironcore.trainers.base_trainer.get_data_iterator', return_value=create_mock_data_iterator()), \
             patch('ironcore.trainers.base_trainer.get_evaluators', return_value=create_mock_evaluators()):
            trainer = LanguageModelTrainer(main_config, forward_step, F.cross_entropy)
            trainer._initialize()

            torch.cuda.reset_peak_memory_stats()
            step_times = []
            losses = []

            for step in range(num_steps):
                step_start = time.time()
                loss, _, _ = trainer.train_step(step=step)
                step_times.append((time.time() - step_start) * 1000)
                losses.append(loss)

            peak_vram_mb = torch.cuda.max_memory_allocated() / 1024**2
            alloc_vram_mb = torch.cuda.memory_allocated() / 1024**2
            avg_step_ms = sum(step_times) / len(step_times)

            trainer._finalize_process()

            return {
                "seq_len": seq_len,
                "peak_vram_mb": peak_vram_mb,
                "alloc_vram_mb": alloc_vram_mb,
                "avg_step_ms": avg_step_ms,
                "final_loss": losses[-1] if losses else None,
                "status": "OK",
            }

    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            return {"seq_len": seq_len, "status": "OOM"}
        return {"seq_len": seq_len, "status": "FAIL", "error": str(e)[:150]}
    except Exception as e:
        return {"seq_len": seq_len, "status": "FAIL", "error": str(e)[:150]}


def main():
    print()
    print("=" * 80)
    print("  TRAINING ACTIVATION MEMORY vs CONTEXT LENGTH — 13B (Full Offload)")
    print("=" * 80)
    print(f"  GPU:     {torch.cuda.get_device_name()} ({torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB)")
    print(f"  Model:   ~13B (40 layers, GQA)")
    print(f"  Offload: M1(bf16) + M2 + M3")
    print("=" * 80)
    print()
    print(f"  {'Seq Len':>10s} {'Peak VRAM':>12s} {'Alloc VRAM':>12s} {'Avg Step':>12s} {'Throughput':>14s} {'Loss':>10s} {'Status':>8s}")
    print("  " + "-" * 86)

    results = []

    for seq_len in SEQ_LENS:
        r = run_training_steps(seq_len, num_steps=3)
        results.append(r)

        if r["status"] == "OK":
            throughput = 60 / (r["avg_step_ms"] / 1000)
            print(f"  {seq_len:>10d} {r['peak_vram_mb']:>10.1f}MB {r['alloc_vram_mb']:>10.1f}MB "
                  f"{r['avg_step_ms']:>10.0f}ms {throughput:>12.1f}/min "
                  f"{r['final_loss']:>10.4f} {r['status']:>8s}")
        else:
            err = r.get("error", "")
            print(f"  {seq_len:>10d} {'N/A':>12s} {'N/A':>12s} {'N/A':>12s} {'N/A':>14s} {'N/A':>10s} {r['status']:>8s} {err}")
            if "OOM" in r["status"]:
                break

    # Analysis
    print()
    print("=" * 80)
    print("  ANALYSIS")
    print("=" * 80)

    ok = [r for r in results if r["status"] == "OK"]
    if len(ok) >= 2:
        # Activation memory scaling
        r1, r2 = ok[0], ok[-1]
        ratio = r2["seq_len"] / r1["seq_len"]
        vram_ratio = r2["alloc_vram_mb"] / r1["alloc_vram_mb"] if r1["alloc_vram_mb"] > 0 else 0
        time_ratio = r2["avg_step_ms"] / r1["avg_step_ms"] if r1["avg_step_ms"] > 0 else 0

        print(f"  Context {r1['seq_len']}→{r2['seq_len']} ({ratio:.0f}x):")
        print(f"    VRAM scaling:    {vram_ratio:.2f}x (ideal linear = {ratio:.0f}x)")
        print(f"    Step time scale: {time_ratio:.2f}x")
        print()

        # Activation per token estimate
        if len(ok) >= 2:
            # Activation memory ≈ VRAM increase proportional to seq_len
            # Slope of (alloc_vram vs seq_len) gives per-token activation cost
            import numpy as np
            xs = [r["seq_len"] for r in ok]
            ys = [r["alloc_vram_mb"] for r in ok]
            slope, _ = np.polyfit(xs, ys, 1)
            print(f"  Activation memory per 1K tokens: ~{slope:.1f} MB")
            print()
            print("  Extrapolated training VRAM:")
            for sl in [8192, 16384, 32768]:
                est_vram = slope * sl / 1024  # MB per K tokens
                fits = "✅ fits" if est_vram + 2 < 24 * 1024 else "❌ too large"
                print(f"    seq_len={sl:>6d}: ~{est_vram/1024:.1f} GB steady-state  {fits}")


if __name__ == "__main__":
    main()
