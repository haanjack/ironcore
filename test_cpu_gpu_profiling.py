#!/usr/bin/env python3
"""CPU vs GPU utilization profiling — 13B full offload.

Measures how CPU AdamW bottleneck changes with seq_len.
Longer sequences shift compute to GPU, improving overlap.

Usage:
    python test_cpu_gpu_profiling.py
"""
import os, sys, time, torch, threading
import psutil
from pathlib import Path

os.environ.setdefault("MASTER_ADDR", "localhost")
os.environ.setdefault("MASTER_PORT", "29507")
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


class ResourceMonitor:
    """Samples CPU and GPU utilization in background thread."""

    def __init__(self, interval_ms=200):
        self.interval_s = interval_ms / 1000
        self.samples = []
        self._running = False
        self._thread = None

    def _loop(self):
        proc = psutil.Process(os.getpid())
        while self._running:
            cpu = proc.cpu_percent()
            gpu = torch.cuda.utilization()
            vram = torch.cuda.memory_allocated() / 1024**2
            self.samples.append((time.time(), cpu, gpu, vram))
            time.sleep(self.interval_s)

    def start(self):
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=2)

    def summary(self):
        if not self.samples:
            return {}
        cpus = [s[1] for s in self.samples]
        gpus = [s[2] for s in self.samples]
        return {
            "cpu_avg": sum(cpus) / len(cpus),
            "cpu_max": max(cpus),
            "gpu_avg": sum(gpus) / len(gpus),
            "gpu_max": max(gpus),
        }


def run_profile(seq_len, num_steps=3):
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
                               model_path="/tmp/validate_prof", log_interval=1,
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

            monitor = ResourceMonitor(interval_ms=200)
            step_times = []

            torch.cuda.reset_peak_memory_stats()
            monitor.start()
            # warmup
            trainer.train_step(step=0)

            for step in range(1, num_steps + 1):
                t0 = time.time()
                loss, _, _ = trainer.train_step(step=step)
                step_times.append((time.time() - t0) * 1000)

            monitor.stop()
            peak_vram = torch.cuda.max_memory_allocated() / 1024**2
            res = monitor.summary()
            trainer._finalize_process()

            return {
                "seq_len": seq_len, "peak_vram_mb": peak_vram,
                "avg_step_ms": sum(step_times)/len(step_times),
                "final_loss": loss, "cpu_avg": res["cpu_avg"],
                "cpu_max": res["cpu_max"], "gpu_avg": res["gpu_avg"],
                "gpu_max": res["gpu_max"], "status": "OK",
            }
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            return {"seq_len": seq_len, "status": "OOM"}
        return {"seq_len": seq_len, "status": "FAIL", "error": str(e)[:200]}
    except Exception as e:
        return {"seq_len": seq_len, "status": "FAIL", "error": str(e)[:200]}


def main():
    print()
    print("=" * 80)
    print("  CPU vs GPU UTILIZATION PROFILING — 13B Full Offload")
    print("=" * 80)
    print(f"  GPU: {torch.cuda.get_device_name()} "
          f"({torch.cuda.get_device_properties(0).total_memory/1024**3:.1f} GB)")
    print(f"  CPU: {psutil.cpu_count(logical=False)}P / {psutil.cpu_count()}T")
    print(f"  RAM: {psutil.virtual_memory().total/1024**3:.0f} GB")
    print(f"  Model: ~13B full offload (optimizer_offload+weight_offload+activation_spill, bf16 optim)")
    print("=" * 80)
    print()

    seq_lens = [512, 1024, 2048, 4096]
    results = []

    print(f"  {'Seq':>6s} {'AvgStep':>10s} {'CPUavg':>8s} {'CPUmax':>8s} "
          f"{'GPUavg':>8s} {'GPUmax':>8s} {'PeakVRAM':>10s} {'Bottleneck':>18s}")
    print("  " + "-" * 80)

    for sl in seq_lens:
        r = run_profile(sl, num_steps=3)
        results.append(r)
        if r["status"] == "OK":
            if r["cpu_avg"] > 70 and r["gpu_avg"] < 30:
                bn = "CPU-bound (AdamW)"
            elif r["gpu_avg"] > 50 and r["cpu_avg"] < 40:
                bn = "GPU-bound (attn/MLP)"
            else:
                bn = "Mixed (overlapping)"
            print(f"  {sl:>6d} {r['avg_step_ms']:>8.0f}ms {r['cpu_avg']:>7.1f}% "
                  f"{r['cpu_max']:>7.1f}% {r['gpu_avg']:>7.1f}% "
                  f"{r['gpu_max']:>7.1f}% {r['peak_vram_mb']:>8.1f}MB {bn}")
        else:
            err = r.get("error", "")
            print(f"  {sl:>6d} {'N/A':>10s} {'N/A':>8s} {'N/A':>8s} "
                  f"{'N/A':>8s} {'N/A':>8s} {'N/A':>10s} {r['status']} {err[:40]}")
            if "OOM" in r["status"]:
                break

    ok = [r for r in results if r["status"] == "OK"]
    if len(ok) >= 2:
        print()
        print("=" * 80)
        print("  GPU UTILIZATION TREND (should rise with longer sequences)")
        print("=" * 80)
        for r in ok:
            blen = int(r["gpu_avg"] / 2)
            bar = "█" * blen + "░" * (50 - blen)
            print(f"  seq={r['seq_len']:>5d} |{bar}| {r['gpu_avg']:.1f}%")

        print()
        print("  CPU UTILIZATION TREND (should fall as GPU takes more work)")
        print("=" * 80)
        for r in ok:
            blen = int(r["cpu_avg"] / 2)
            bar = "█" * blen + "░" * (50 - blen)
            print(f"  seq={r['seq_len']:>5d} |{bar}| {r['cpu_avg']:.1f}%")

        if len(ok) >= 2:
            r0, rn = ok[0], ok[-1]
            ratio = rn["seq_len"] / r0["seq_len"]
            gpu_gain = rn["gpu_avg"] - r0["gpu_avg"]
            cpu_drop = r0["cpu_avg"] - rn["cpu_avg"]
            print()
            print(f"  seq_len {r0['seq_len']}→{rn['seq_len']} ({ratio:.0f}x longer):")
            print(f"    GPU utilization: +{gpu_gain:.1f}pp "
                  f"({r0['gpu_avg']:.1f}% → {rn['gpu_avg']:.1f}%)")
            print(f"    CPU utilization: {cpu_drop:+.1f}pp "
                  f"({r0['cpu_avg']:.1f}% → {rn['cpu_avg']:.1f}%)")


if __name__ == "__main__":
    main()
