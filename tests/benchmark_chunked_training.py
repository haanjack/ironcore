#!/usr/bin/env python
"""
Training benchmark for chunked tensor parallelism.

Measures iteration time and peak memory across:
  - Sequence lengths: 1024, 4096, 8192
  - Chunk sizes: None (baseline), 512, 256
  - Modes: eager and torch.compile (inductor)
  - Uses flash attention + bfloat16

Usage:
    python tests/benchmark_chunked_training.py
"""

import gc
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ironcore.config import (
    DataConfig,
    InitConfig,
    MainConfig,
    ModelConfig,
    OperationConfig,
    OptimConfig,
    ParallelConfig,
    TrainerConfig,
    UtilsConfig,
)
from ironcore.models.transformer import TransformerModel
from ironcore.parallel.parallel_states import initialize_model_parallel

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ── Model ──────────────────────────────────────────────────────────────
D_MODEL = 512
NUM_HEADS = 8
HEAD_DIM = D_MODEL // NUM_HEADS  # 64
NUM_GROUPS = 8
D_FFN = 2048
NUM_LAYERS = 6
BATCH_SIZE = 1

# ── Benchmark ──────────────────────────────────────────────────────────
EAGER_WARMUP = 3
COMPILE_WARMUP = 5  # torch.compile needs more warmup for compilation
MEASURE_STEPS = 10

SEQUENCE_LENGTHS = [1024, 4096, 8192]
CHUNK_SIZES = [None, 2048, 1024, 512, 256]  # None = baseline (no chunking)


def create_config(seq_len, chunk_size):
    return MainConfig(
        model=ModelConfig(
            d_model=D_MODEL,
            num_attention_heads=NUM_HEADS,
            num_attention_groups=NUM_GROUPS,
            head_dim=HEAD_DIM,
            d_ffn=D_FFN,
            num_layers=NUM_LAYERS,
            max_seq_len=seq_len,
            max_position_embeddings=seq_len,
            dropout_attn=0.0,
            dropout_mlp=0.0,
            dropout_embd=0.0,
            no_bias=False,
            precision="bfloat16",
        ),
        trainer=TrainerConfig(
            tensor_model_parallel_size=1,
            use_flash_attn=True,
            sequence_chunk_size=chunk_size,
        ),
        init=InitConfig(seed=42, init_std=0.02),
        optim=OptimConfig(),
        data=DataConfig(),
        parallel=ParallelConfig(),
        operation=OperationConfig(),
        utils=UtilsConfig(),
    )


def benchmark_one(seq_len, chunk_size, device, use_compile):
    """Benchmark a single (seq_len, chunk_size, compile) configuration."""
    config = create_config(seq_len, chunk_size)

    torch.manual_seed(42)
    torch.cuda.manual_seed(42)

    model = TransformerModel(config).to(device=device, dtype=torch.bfloat16)
    model.init_weights()
    model.train()

    if use_compile:
        torch._dynamo.reset()
        model = torch.compile(model, backend="inductor", mode="default")

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    # Input — no explicit mask needed for flash attention (causal=True internally)
    hidden = torch.randn(BATCH_SIZE, seq_len, D_MODEL, device=device, dtype=torch.bfloat16)

    # ── Warmup ─────────────────────────────────────────────────────────
    warmup = COMPILE_WARMUP if use_compile else EAGER_WARMUP
    for _ in range(warmup):
        optimizer.zero_grad()
        output = model(hidden, attention_mask=None, rotary_pos_emb=None)
        loss = output.pow(2).mean()
        loss.backward()
        optimizer.step()

    torch.cuda.synchronize()

    # ── Reset memory stats after warmup ────────────────────────────────
    torch.cuda.reset_peak_memory_stats(device)

    # ── Measure ────────────────────────────────────────────────────────
    times = []
    last_loss = None
    for _ in range(MEASURE_STEPS):
        torch.cuda.synchronize()
        t0 = time.perf_counter()

        optimizer.zero_grad()
        output = model(hidden, attention_mask=None, rotary_pos_emb=None)
        loss = output.pow(2).mean()
        loss.backward()
        optimizer.step()

        torch.cuda.synchronize()
        t1 = time.perf_counter()
        times.append(t1 - t0)
        last_loss = loss.item()

    peak_alloc = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
    peak_resv = torch.cuda.max_memory_reserved(device) / (1024 ** 2)

    # ── Cleanup ────────────────────────────────────────────────────────
    del model, optimizer, hidden, output, loss
    torch._dynamo.reset()
    torch.cuda.empty_cache()
    gc.collect()

    return {
        "seq_len": seq_len,
        "chunk_size": chunk_size,
        "compile": use_compile,
        "avg_time": sum(times) / len(times),
        "min_time": min(times),
        "max_time": max(times),
        "std_time": (sum((t - sum(times) / len(times)) ** 2 for t in times) / len(times)) ** 0.5,
        "peak_alloc_mib": peak_alloc,
        "peak_resv_mib": peak_resv,
        "last_loss": last_loss,
    }


def nan_result(seq_len, chunk_size, use_compile):
    return {
        "seq_len": seq_len,
        "chunk_size": chunk_size,
        "compile": use_compile,
        "avg_time": float("nan"),
        "min_time": float("nan"),
        "max_time": float("nan"),
        "std_time": float("nan"),
        "peak_alloc_mib": float("nan"),
        "peak_resv_mib": float("nan"),
        "last_loss": float("nan"),
    }


def print_section(results, title):
    """Print results table and comparison for one mode (eager or compiled)."""
    print(f"\n{'=' * 105}")
    print(f"  {title}")
    print(f"{'=' * 105}")

    header = (
        f"{'Seq Len':>8}  {'Chunk':>6}  "
        f"{'Avg (s)':>9}  {'Min (s)':>9}  {'Max (s)':>9}  {'Std (s)':>9}  "
        f"{'Alloc (MiB)':>12}  {'Resv (MiB)':>12}  {'Loss':>12}"
    )
    print(header)
    print("-" * 105)

    baselines = {}
    for r in results:
        cs_label = "none" if r["chunk_size"] is None else str(r["chunk_size"])
        if r["chunk_size"] is None:
            baselines[r["seq_len"]] = r
        print(
            f"{r['seq_len']:>8}  {cs_label:>6}  "
            f"{r['avg_time']:>9.4f}  {r['min_time']:>9.4f}  {r['max_time']:>9.4f}  {r['std_time']:>9.4f}  "
            f"{r['peak_alloc_mib']:>12.1f}  {r['peak_resv_mib']:>12.1f}  {r['last_loss']:>12.6f}"
        )

    print(f"\n  {'Seq Len':>8}  {'Chunk':>6}  {'Speedup':>10}  {'Time Delta':>12}  {'Mem Delta (MiB)':>16}")
    print(f"  {'-' * 70}")
    for r in results:
        if r["chunk_size"] is None:
            continue
        base = baselines.get(r["seq_len"])
        if base is None or base["avg_time"] != base["avg_time"]:
            continue
        if r["avg_time"] != r["avg_time"]:
            print(f"  {r['seq_len']:>8}  {r['chunk_size']:>6}  {'OOM':>10}")
            continue
        speedup = base["avg_time"] / r["avg_time"]
        time_delta = r["avg_time"] - base["avg_time"]
        mem_delta = r["peak_alloc_mib"] - base["peak_alloc_mib"]
        print(
            f"  {r['seq_len']:>8}  {r['chunk_size']:>6}  "
            f"{speedup:>9.3f}x  "
            f"{time_delta:>+11.4f}s  "
            f"{mem_delta:>+15.1f}"
        )


def main():
    try:
        from flash_attn import flash_attn_varlen_func  # noqa: F401
    except ImportError:
        print("ERROR: flash_attn is required for this benchmark.")
        sys.exit(1)

    device = torch.device("cuda")

    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", "12355")
    if not torch.distributed.is_initialized():
        torch.distributed.init_process_group(backend="nccl", rank=0, world_size=1)
    try:
        initialize_model_parallel(tensor_model_parallel_size=1, timeout_in_minutes=1.0)
    except Exception:
        pass

    gpu_name = torch.cuda.get_device_properties(0).name
    gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)

    print(f"GPU: {gpu_name} ({gpu_mem:.1f} GB)")
    print(f"Model: d={D_MODEL}, heads={NUM_HEADS}, ffn={D_FFN}, layers={NUM_LAYERS}")
    print(f"Batch size: {BATCH_SIZE}, dtype: bfloat16, flash_attn: True")
    print(f"Warmup: eager={EAGER_WARMUP}, compiled={COMPILE_WARMUP} steps | Measure: {MEASURE_STEPS} steps")

    # ── Phase 1: Eager mode ────────────────────────────────────────────
    eager_results = []
    print("\n── Phase 1: Eager mode ──")
    for seq_len in SEQUENCE_LENGTHS:
        for chunk_size in CHUNK_SIZES:
            label = f"seq={seq_len:<5} chunk={'none' if chunk_size is None else chunk_size:<5}"
            print(f"  {label} ...", end=" ", flush=True)
            try:
                r = benchmark_one(seq_len, chunk_size, device, use_compile=False)
                eager_results.append(r)
                print(f"avg={r['avg_time']:.4f}s  mem={r['peak_alloc_mib']:.0f} MiB")
            except torch.cuda.OutOfMemoryError:
                print("OOM")
                torch.cuda.empty_cache()
                gc.collect()
                eager_results.append(nan_result(seq_len, chunk_size, False))

    # ── Phase 2: Compiled mode ─────────────────────────────────────────
    compiled_results = []
    print("\n── Phase 2: torch.compile (inductor, mode=default) ──")
    for seq_len in SEQUENCE_LENGTHS:
        for chunk_size in CHUNK_SIZES:
            label = f"seq={seq_len:<5} chunk={'none' if chunk_size is None else chunk_size:<5}"
            print(f"  {label} ...", end=" ", flush=True)
            try:
                r = benchmark_one(seq_len, chunk_size, device, use_compile=True)
                compiled_results.append(r)
                print(f"avg={r['avg_time']:.4f}s  mem={r['peak_alloc_mib']:.0f} MiB")
            except torch.cuda.OutOfMemoryError:
                print("OOM")
                torch.cuda.empty_cache()
                gc.collect()
                compiled_results.append(nan_result(seq_len, chunk_size, True))

    # ── Print results ──────────────────────────────────────────────────
    print_section(eager_results, "EAGER MODE")
    print_section(compiled_results, "TORCH.COMPILE MODE (inductor, default)")

    # ── Cross-mode comparison ──────────────────────────────────────────
    print(f"\n{'=' * 90}")
    print("  COMPILE vs EAGER (same chunk config)")
    print(f"{'=' * 90}")
    print(f"{'Seq Len':>8}  {'Chunk':>6}  {'Eager (s)':>10}  {'Compiled (s)':>13}  {'Compile Speedup':>16}  {'Mem Δ (MiB)':>12}")
    print("-" * 90)

    for e, c in zip(eager_results, compiled_results, strict=True):
        cs_label = "none" if e["chunk_size"] is None else str(e["chunk_size"])
        if e["avg_time"] != e["avg_time"] or c["avg_time"] != c["avg_time"]:
            print(f"{e['seq_len']:>8}  {cs_label:>6}  {'—':>10}  {'—':>13}  {'—':>16}  {'—':>12}")
            continue
        speedup = e["avg_time"] / c["avg_time"]
        mem_delta = c["peak_alloc_mib"] - e["peak_alloc_mib"]
        print(
            f"{e['seq_len']:>8}  {cs_label:>6}  "
            f"{e['avg_time']:>10.4f}  {c['avg_time']:>13.4f}  "
            f"{speedup:>15.3f}x  "
            f"{mem_delta:>+11.1f}"
        )

    print()


if __name__ == "__main__":
    main()
