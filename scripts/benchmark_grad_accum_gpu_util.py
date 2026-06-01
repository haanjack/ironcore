#!/usr/bin/env python3
"""
13B Grad Accum GPU Utilization Benchmark.

Measures how gradient accumulation affects GPU utilization with full offload.
Hypothesis: high grad_accum amortizes CPU optimizer bottleneck, raising GPU util.

Usage:
    python scripts/benchmark_grad_accum_gpu_util.py
"""

import os
import shutil
import sys
import threading
import time
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent.parent))

GRAD_ACCUM_VALUES = [1, 8, 64, 128]
STEPS_PER_CONFIG = 3  # 1 warmup + 2 measured
SEQ_LEN = 512

LLAMA_13B = {
    "d_model": 5120,
    "d_ffn": 13824,
    "num_layers": 40,
    "num_attention_heads": 40,
    "num_attention_groups": 8,
    "head_dim": 128,
    "max_seq_len": SEQ_LEN,
    "precision": "bfloat16",
    "ln_type": "rmsnorm",
    "ln_eps": 1e-5,
    "activation_type": "swiglu",
    "positional_embedding": {"type": "rope", "base": 10000},
}


class CPUMonitor:
    """Background CPU utilization sampler."""

    def __init__(self):
        self.samples = []
        self._stop = threading.Event()

    def start(self):
        self._stop.clear()
        self.samples.clear()
        self._thread = threading.Thread(target=self._poll, daemon=True)
        self._thread.start()

    def stop(self) -> float:
        self._stop.set()
        self._thread.join(timeout=5)
        return sum(self.samples) / len(self.samples) if self.samples else 0.0

    def _poll(self):
        import psutil

        while not self._stop.is_set():
            self.samples.append(psutil.cpu_percent(interval=0.5))


def build_config(grad_accum: int) -> dict:
    """Build full offload config for 13B with given gradient_accumulation_steps."""
    return {
        "model": {**LLAMA_13B},
        "operation": {
            "train_steps": STEPS_PER_CONFIG + 5,
            "activation_recompute": False,
            "no_save": True,
        },
        "trainer": {
            "micro_batch_size": 1,
            "train_batch_size": grad_accum,
            "gradient_accumulation_steps": grad_accum,
            "model_path": f"/tmp/benchmark_grad_accum_{grad_accum}",
            "log_interval": 1,
            "use_flash_attn": True,
        },
        "optim": {
            "optimizer": "adamw",
            "lr_scheduler": "cosine",
            "max_lr": 3e-4,
            "min_lr": 3e-5,
            "warmup_steps": 1,
            "annealing_steps": STEPS_PER_CONFIG,
            "weight_decay": 0.1,
            "clip_grad": 1.0,
        },
        "init": {"seed": 42, "init_std": 0.02},
        "parallel": {
            "world_size": 1,
            "rank": 0,
            "local_rank": 0,
            "dist_backend": "nccl",
            "use_fsdp": False,
        },
        "data": {"seq_length": SEQ_LEN},
        "offload": {
            "enabled": True,
            "optimizer_offload": True,
            "optimizer_state_precision": "bf16",
            "weight_offload": True,
            "weight_prefetch_layers": 2,
            "weight_storage_precision": "bf16",
            "activation_spill": True,
            "activation_spill_granularity": "sub_layer",
            "pinned_memory_pool_gb": 80.0,
        },
        "utils": {"log_level": "WARNING"},
        "profiler": {"gpu_profiler": False, "torch_profiler": False},
        "peft": {"method": "none"},
        "alignment": {"method": "dpo"},
    }


def run_benchmark(grad_accum: int) -> dict:
    """Run benchmark for a single grad_accum value."""
    from unittest.mock import patch

    from tests.integration.offload.conftest import (
        create_mock_data_iterator,
        create_mock_evaluators,
    )

    from ironcore.config import (
        AlignmentConfig,
        DataConfig,
        InitConfig,
        MainConfig,
        ModelConfig,
        OffloadConfig,
        OperationConfig,
        OptimConfig,
        ParallelConfig,
        PEFTConfig,
        ProfilerConfig,
        TrainerConfig,
        UtilsConfig,
    )
    from ironcore.global_vars import reset_global_states
    from ironcore.trainers import LanguageModelTrainer

    config = build_config(grad_accum)

    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", "29500")
    os.environ.setdefault("LOCAL_RANK", "0")
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")

    if not torch.distributed.is_initialized():
        torch.distributed.init_process_group(backend="nccl", rank=0, world_size=1)

    # Cleanup
    model_path = config["trainer"]["model_path"]
    if os.path.exists(model_path):
        shutil.rmtree(model_path, ignore_errors=True)

    reset_global_states()

    mc = config["model"]
    model_cfg = ModelConfig(
        name="llama-13b",
        d_model=mc["d_model"],
        d_ffn=mc["d_ffn"],
        num_layers=mc["num_layers"],
        num_attention_heads=mc["num_attention_heads"],
        num_attention_groups=mc["num_attention_groups"],
        head_dim=mc["head_dim"],
        max_seq_len=mc["max_seq_len"],
        precision=mc["precision"],
        ln_type=mc["ln_type"],
        ln_eps=mc["ln_eps"],
        activation_type=mc["activation_type"],
        vocab_name_or_path="gpt2",
        tokenizer_type="gpt2",
    )
    model_cfg.positional_embedding.type = mc["positional_embedding"]["type"]
    model_cfg.positional_embedding.base = mc["positional_embedding"]["base"]

    tc = config["trainer"]
    trainer_cfg = TrainerConfig(
        micro_batch_size=tc["micro_batch_size"],
        train_batch_size=tc["train_batch_size"],
        gradient_accumulation_steps=tc["gradient_accumulation_steps"],
        model_path=tc["model_path"],
        log_interval=tc["log_interval"],
        use_flash_attn=tc["use_flash_attn"],
    )

    oc = config["operation"]
    oc_ = config["optim"]

    offload_cfg = OffloadConfig()
    for k, v in config["offload"].items():
        if hasattr(offload_cfg, k):
            setattr(offload_cfg, k, v)

    main_config = MainConfig(
        model=model_cfg,
        init=InitConfig(**config["init"]),
        optim=OptimConfig(
            optimizer=oc_["optimizer"],
            lr_scheduler=oc_["lr_scheduler"],
            max_lr=oc_["max_lr"],
            min_lr=oc_["min_lr"],
            warmup_steps=oc_["warmup_steps"],
            annealing_steps=oc_["annealing_steps"],
            weight_decay=oc_["weight_decay"],
            clip_grad=oc_["clip_grad"],
        ),
        data=DataConfig(seq_length=config["data"]["seq_length"]),
        parallel=ParallelConfig(
            world_size=config["parallel"]["world_size"],
            rank=config["parallel"]["rank"],
            local_rank=config["parallel"]["local_rank"],
            dist_backend=config["parallel"]["dist_backend"],
            use_fsdp=config["parallel"]["use_fsdp"],
        ),
        trainer=trainer_cfg,
        operation=OperationConfig(
            train_steps=oc["train_steps"],
            activation_recompute=oc["activation_recompute"],
            no_save=oc["no_save"],
        ),
        utils=UtilsConfig(**config["utils"]),
        profiler=ProfilerConfig(**config["profiler"]),
        peft=PEFTConfig(method=config["peft"]["method"]),
        alignment=AlignmentConfig(method=config["alignment"]["method"]),
        offload=offload_cfg,
    )

    def forward_step(model, _data_iterator):
        device = next(model.parameters()).device
        input_ids = torch.randint(0, 32000, (1, SEQ_LEN), device=device)
        labels = input_ids.clone()
        logits = model(input_ids, labels=None)
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()
        loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        return loss

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
        trainer = LanguageModelTrainer(main_config, forward_step, F.cross_entropy)
        trainer._initialize()

        torch.cuda.reset_peak_memory_stats()

        results = {"grad_accum": grad_accum, "steps": [], "error": None}

        try:
            for step in range(STEPS_PER_CONFIG):
                cpu_monitor = CPUMonitor()
                cpu_monitor.start()

                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)

                torch.cuda.synchronize()
                wall_start = time.perf_counter()
                start_event.record()

                loss, _, _ = trainer.train_step(step=step)

                end_event.record()
                torch.cuda.synchronize()
                wall_end = time.perf_counter()

                cpu_util = cpu_monitor.stop()

                wall_time = wall_end - wall_start
                gpu_time = start_event.elapsed_time(end_event) / 1000.0
                gpu_util = (gpu_time / wall_time) * 100 if wall_time > 0 else 0

                results["steps"].append(
                    {
                        "step": step,
                        "loss": loss,
                        "wall_time_s": round(wall_time, 2),
                        "gpu_time_s": round(gpu_time, 2),
                        "gpu_util_pct": round(gpu_util, 1),
                        "cpu_util_pct": round(cpu_util, 1),
                    }
                )

                label = "warmup" if step == 0 else "measured"
                print(
                    f"    step {step} ({label}): "
                    f"wall={wall_time:.1f}s  gpu={gpu_time:.1f}s  "
                    f"gpu_util={gpu_util:.1f}%  cpu_util={cpu_util:.1f}%  "
                    f"loss={loss:.4f}"
                )

            results["peak_vram_gb"] = round(torch.cuda.max_memory_allocated() / 1024**3, 2)

        except Exception as e:
            results["error"] = str(e)
            print(f"    ERROR: {e}")

        trainer._finalize_process()

    return results


def main():
    print("=" * 70)
    print("  13B Grad Accum GPU Utilization Benchmark")
    print("=" * 70)
    print(f"  GPU: {torch.cuda.get_device_name()}")
    print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    print(f"  Seq len: {SEQ_LEN}")
    print("  Offload: full (optimizer + weight + activation)")
    print(f"  Grad accum values: {GRAD_ACCUM_VALUES}")
    print(f"  Steps per config: {STEPS_PER_CONFIG} (1 warmup + {STEPS_PER_CONFIG - 1} measured)")
    print()

    if not torch.cuda.is_available():
        print("Error: CUDA required")
        sys.exit(1)

    all_results = []

    for ga in GRAD_ACCUM_VALUES:
        print(f"\n--- grad_accum={ga} (GBS={ga}, MBS=1) ---")
        result = run_benchmark(ga)
        all_results.append(result)

    # Summary table
    print("\n" + "=" * 70)
    print("  SUMMARY")
    print("=" * 70)
    print(
        f"\n  {'grad_accum':>11} {'GBS':>5} {'wall(s)':>8} {'gpu(s)':>8} "
        f"{'GPU%':>7} {'CPU%':>7} {'VRAM(GB)':>9}"
    )
    print("  " + "-" * 60)

    for r in all_results:
        if r["error"]:
            print(f"  {r['grad_accum']:>11}  ERROR: {r['error'][:50]}")
            continue

        # Average measured steps (skip warmup)
        measured = [s for s in r["steps"] if s["step"] > 0]
        if not measured:
            continue

        avg_wall = sum(s["wall_time_s"] for s in measured) / len(measured)
        avg_gpu = sum(s["gpu_time_s"] for s in measured) / len(measured)
        avg_gpu_util = sum(s["gpu_util_pct"] for s in measured) / len(measured)
        avg_cpu = sum(s["cpu_util_pct"] for s in measured) / len(measured)
        vram = r.get("peak_vram_gb", 0)

        print(
            f"  {r['grad_accum']:>11} {r['grad_accum']:>5} {avg_wall:>8.1f} {avg_gpu:>8.1f} "
            f"{avg_gpu_util:>6.1f}% {avg_cpu:>6.1f}% {vram:>9.2f}"
        )


if __name__ == "__main__":
    main()
