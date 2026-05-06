#!/usr/bin/env python3
"""CPU thread scaling test — 13B full offload.

Tests whether increasing OpenMP threads improves step time.
Currently at ~880% (8-9 cores), tries up to 24 threads.
"""
import os, sys, time, torch
from pathlib import Path

os.environ.setdefault("MASTER_ADDR", "localhost")
os.environ.setdefault("MASTER_PORT", "29508")
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


SEQ_LEN = 1024
NUM_STEPS = 3


def run_with_threads(num_threads, seq_len=SEQ_LEN, num_steps=NUM_STEPS):
    torch.set_num_threads(num_threads)
    os.environ["OMP_NUM_THREADS"] = str(num_threads)
    os.environ["MKL_NUM_THREADS"] = str(num_threads)
    reset_global_states()

    model_cfg = ModelConfig(
        name="llama-13b", d_model=5120, d_ffn=13824, num_layers=40,
        num_attention_heads=40, num_attention_groups=8, head_dim=128,
        max_seq_len=seq_len, precision="bfloat16", ln_type="rmsnorm",
        activation_type="swiglu", vocab_name_or_path="gpt2",
    )
    model_cfg.positional_embedding.type = "rope"
    model_cfg.positional_embedding.base = 10000

    offload_cfg = OffloadConfig(
        enabled=True, optimizer_offload=True, optimizer_state_precision="bf16",
        weight_offload=True, weight_prefetch_layers=2, weight_storage_precision="bf16",
        activation_spill=True, activation_spill_granularity="sub_layer",
        pinned_memory_pool_gb=80.0, gpu_staging_pool_mb=0.0,
    )

    main_config = MainConfig(
        model=model_cfg,
        init=InitConfig(seed=42),
        optim=OptimConfig(optimizer="adamw", max_lr=3e-4, min_lr=3e-5,
                         warmup_steps=1, annealing_steps=num_steps+1,
                         weight_decay=0.1, clip_grad=1.0),
        data=DataConfig(seq_length=seq_len),
        parallel=ParallelConfig(world_size=1, rank=0, local_rank=0),
        trainer=TrainerConfig(micro_batch_size=1, train_batch_size=1,
                               gradient_accumulation_steps=1,
                               model_path="/tmp/validate_threads", log_interval=1,
                               use_flash_attn=True),
        operation=OperationConfig(train_steps=num_steps+2,
                                   activation_recompute=False, no_save=True),
        utils=UtilsConfig(log_level="WARNING"), profiler=ProfilerConfig(),
        peft=PEFTConfig(method="none"), alignment=AlignmentConfig(),
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
            shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        return loss

    try:
        with patch('ironcore.trainers.base_trainer.get_data_iterator',
                   return_value=create_mock_data_iterator()), \
             patch('ironcore.trainers.base_trainer.get_evaluators',
                   return_value=create_mock_evaluators()):
            trainer = LanguageModelTrainer(main_config, forward_step, F.cross_entropy)
            trainer._initialize()

            # warmup
            trainer.train_step(step=0)

            step_times = []
            for step in range(1, num_steps + 1):
                t0 = time.time()
                loss, _, _ = trainer.train_step(step=step)
                step_times.append((time.time() - t0) * 1000)

            trainer._finalize_process()

            return {
                "threads": num_threads,
                "avg_step_ms": sum(step_times) / len(step_times),
                "min_step_ms": min(step_times),
                "max_step_ms": max(step_times),
                "status": "OK",
            }
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            return {"threads": num_threads, "status": "OOM"}
        return {"threads": num_threads, "status": "FAIL", "error": str(e)[:200]}
    except Exception as e:
        return {"threads": num_threads, "status": "FAIL", "error": str(e)[:200]}


def main():
    import psutil

    print()
    print("=" * 70)
    print("  CPU THREAD SCALING — 13B Full Offload (DDR5)")
    print("=" * 70)
    print(f"  GPU: {torch.cuda.get_device_name()} "
          f"({torch.cuda.get_device_properties(0).total_memory/1024**3:.1f} GB)")
    print(f"  CPU: {psutil.cpu_count(logical=False)}P / {psutil.cpu_count()}T")
    print(f"  RAM: {psutil.virtual_memory().total/1024**3:.0f} GB")
    print(f"  Model: ~13B full offload, seq_len={SEQ_LEN}")
    print("=" * 70)
    print()

    thread_counts = [4, 8, 12, 16, 20, 24]
    results = []

    print(f"  {'Threads':>8s} {'Avg Step':>10s} {'Min':>10s} {'Max':>10s} {'vs 12T':>8s} {'Status':>8s}")
    print("  " + "-" * 60)

    baseline = None
    for t in thread_counts:
        print(f"  Testing {t} threads...", end="", flush=True)
        r = run_with_threads(t)
        results.append(r)
        if r["status"] == "OK":
            if baseline is None:
                baseline = r["avg_step_ms"]
                speedup = "1.00x (ref)"
            else:
                speedup = f"{baseline / r['avg_step_ms']:.2f}x"
            print(f"\r  {t:>8d} {r['avg_step_ms']:>8.0f}ms {r['min_step_ms']:>8.0f}ms "
                  f"{r['max_step_ms']:>8.0f}ms {speedup:>10s} {r['status']}")
        else:
            err = r.get("error", "")[:30]
            print(f"\r  {t:>8d} {'N/A':>10s} {'N/A':>10s} {'N/A':>10s} "
                  f"{'N/A':>8s} {r['status']} {err}")

    ok = [r for r in results if r["status"] == "OK"]
    if len(ok) >= 2:
        print()
        print("  Thread scaling bar chart:")
        best = min(ok, key=lambda r: r["avg_step_ms"])
        for r in ok:
            blen = int(best["avg_step_ms"] / r["avg_step_ms"] * 40)
            bar = "#" * blen + "-" * (40 - blen)
            print(f"  {r['threads']:>3d}T |{bar}| {r['avg_step_ms']:.0f}ms")


if __name__ == "__main__":
    main()
