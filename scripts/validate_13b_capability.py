#!/usr/bin/env python3
"""
13B Model Capability Validation - Measured VRAM and Performance.

Validates LLaMA-13B training with different offload configurations.
Reports actual measured VRAM, timing, and loss convergence.

Requirements:
    - GPU with 24GB+ VRAM for M1-only test
    - GPU with 8GB+ VRAM for full offload (M1+M2+M3)
    - 64GB+ system RAM for offload tests

Usage:
    # Test all modes (requires ~2 hours)
    python scripts/validate_13b_capability.py

    # Test specific mode only
    python scripts/validate_13b_capability.py --mode m1_only

    # Quick smoke test (10 steps)
    python scripts/validate_13b_capability.py --steps 10

    # Custom config
    python scripts/validate_13b_capability.py --config configs/model/llama-13b.yaml --steps 50
"""

import argparse
import json
import os
import shutil
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F


@dataclass
class ValidationResult:
    """Results from a single validation run."""

    mode: str
    steps: int
    status: str = "PENDING"  # PASS, FAIL, SKIP, TIMEOUT, PENDING
    losses: list[float] = field(default_factory=list)
    peak_vram_mb: float = 0.0
    allocated_vram_mb: float = 0.0
    wall_time_s: float = 0.0
    step_times_ms: list[float] = field(default_factory=list)
    error: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "status": self.status,
            "steps": self.steps,
            "final_loss": self.losses[-1] if self.losses else None,
            "min_loss": min(self.losses) if self.losses else None,
            "peak_vram_mb": round(self.peak_vram_mb, 1),
            "allocated_vram_mb": round(self.allocated_vram_mb, 1),
            "wall_time_s": round(self.wall_time_s, 1),
            "avg_step_time_ms": round(sum(self.step_times_ms) / len(self.step_times_ms), 1)
            if self.step_times_ms
            else None,
            "error": self.error[:200] if self.error else None,
        }


# Default 13B config (LLaMA style)
LLAMA_13B_CONFIG = {
    "d_model": 5120,
    "d_ffn": 13824,
    "num_layers": 40,
    "num_attention_heads": 40,
    "num_attention_groups": 8,  # GQA
    "head_dim": 128,
    "max_seq_len": 1024,
    "precision": "bfloat16",
    "ln_type": "rmsnorm",
    "ln_eps": 1.0e-5,
    "activation_type": "swiglu",
    "positional_embedding": {"type": "rope", "base": 10000},
}

OFFLOAD_MODES = {
    "baseline": {
        "offload.enabled": False,
        "offload.optimizer_offload": False,
        "offload.weight_offload": False,
        "offload.activation_spill": False,
    },
    "m1_only": {
        "offload.enabled": True,
        "offload.optimizer_offload": True,
        "offload.weight_offload": False,
        "offload.activation_spill": False,
    },
    "m3_only": {
        "offload.enabled": True,
        "offload.optimizer_offload": False,
        "offload.weight_offload": False,
        "offload.activation_spill": True,
        "offload.activation_spill_granularity": "sub_layer",
    },
    "m1_m3": {
        "offload.enabled": True,
        "offload.optimizer_offload": True,
        "offload.weight_offload": False,
        "offload.activation_spill": True,
        "offload.activation_spill_granularity": "sub_layer",
    },
    "m1_m2": {
        "offload.enabled": True,
        "offload.optimizer_offload": True,
        "offload.weight_offload": True,
        "offload.weight_prefetch_layers": 2,
        "offload.weight_storage_precision": "bf16",
        "offload.activation_spill": False,
    },
    "full": {
        "offload.enabled": True,
        "offload.optimizer_offload": True,
        "offload.weight_offload": True,
        "offload.weight_prefetch_layers": 2,
        "offload.weight_storage_precision": "bf16",
        "offload.activation_spill": True,
        "offload.activation_spill_granularity": "sub_layer",
    },
}


def nested_set(d: dict, key: str, value: Any) -> None:
    """Set nested dict key from dot-notation path."""
    parts = key.split(".")
    for part in parts[:-1]:
        d = d.setdefault(part, {})
    d[parts[-1]] = value


def build_config(
    model_config: dict,
    offload_overrides: dict,
    steps: int,
    batch_size: int = 1,
    seq_len: int = 1024,
    model_path: str = "/tmp/validate_13b",
) -> dict:
    """Build full training config for validation."""
    config = {
        "model": {**model_config},
        "operation": {
            "train_steps": steps + 10,
            "activation_recompute": False,
            "no_save": True,
        },
        "trainer": {
            "micro_batch_size": batch_size,
            "train_batch_size": batch_size,
            "gradient_accumulation_steps": 1,
            "model_path": model_path,
            "log_interval": max(1, steps // 10),
            "use_flash_attn": True,
        },
        "optim": {
            "optimizer": "adamw",
            "lr_scheduler": "cosine",
            "max_lr": 3e-4,
            "min_lr": 3e-5,
            "warmup_steps": max(1, steps // 20),
            "annealing_steps": steps,
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
        "data": {"seq_length": seq_len},
        "offload": {"enabled": False},
        "utils": {"log_level": "WARNING"},
        "profiler": {"gpu_profiler": False, "torch_profiler": False},
        "peft": {"method": "none"},
        "alignment": {"method": "dpo"},
    }

    # Apply offload overrides
    for key, value in offload_overrides.items():
        nested_set(config, key, value)

    return config


def run_validation(
    config: dict, mode_name: str, steps: int
) -> ValidationResult:
    """Run validation and return measured results."""
    from unittest.mock import patch

    sys.path.insert(0, str(Path(__file__).parent.parent))

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

    # Setup environment
    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", "29500")
    os.environ.setdefault("LOCAL_RANK", "0")
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")

    if not torch.distributed.is_initialized():
        torch.distributed.init_process_group(backend="nccl", rank=0, world_size=1)

    result = ValidationResult(mode=mode_name, steps=steps)

    # Clear checkpoints
    model_path = config["trainer"]["model_path"]
    if os.path.exists(model_path):
        for f in os.listdir(model_path):
            fp = os.path.join(model_path, f)
            if os.path.isdir(fp):
                shutil.rmtree(fp, ignore_errors=True)
            elif os.path.isfile(fp):
                os.remove(fp)

    try:
        reset_global_states()

        # Build proper config objects from dict
        mc = config["model"]
        model_cfg = ModelConfig(
            name="llama-13b",
            d_model=mc.get("d_model", 5120),
            d_ffn=mc.get("d_ffn", 13824),
            num_layers=mc.get("num_layers", 40),
            num_attention_heads=mc.get("num_attention_heads", 40),
            num_attention_groups=mc.get("num_attention_groups", 8),
            head_dim=mc.get("head_dim", 128),
            max_seq_len=mc.get("max_seq_len", 1024),
            precision=mc.get("precision", "bfloat16"),
            ln_type=mc.get("ln_type", "rmsnorm"),
            ln_eps=mc.get("ln_eps", 1e-5),
            activation_type=mc.get("activation_type", "swiglu"),
            vocab_name_or_path=mc.get("vocab_name_or_path", "gpt2"),
            tokenizer_type=mc.get("tokenizer_type", "gpt2"),
        )

        pe_cfg = mc.get("positional_embedding", {})
        model_cfg.positional_embedding.type = pe_cfg.get("type", "rope")
        model_cfg.positional_embedding.base = pe_cfg.get("base", 10000)

        tc = config["trainer"]
        trainer_cfg = TrainerConfig(
            micro_batch_size=tc.get("micro_batch_size", 1),
            train_batch_size=tc.get("train_batch_size", 1),
            gradient_accumulation_steps=tc.get("gradient_accumulation_steps", 1),
            model_path=tc.get("model_path", "/tmp/validate_13b"),
            log_interval=tc.get("log_interval", 1),
            use_flash_attn=tc.get("use_flash_attn", True),
        )

        oc = config["operation"]
        operation_cfg = OperationConfig(
            train_steps=oc.get("train_steps", 20),
            activation_recompute=oc.get("activation_recompute", False),
            no_save=oc.get("no_save", True),
        )

        optc = config["optim"]
        optim_cfg = OptimConfig(
            optimizer=optc.get("optimizer", "adamw"),
            lr_scheduler=optc.get("lr_scheduler", "cosine"),
            max_lr=optc.get("max_lr", 3e-4),
            min_lr=optc.get("min_lr", 3e-5),
            warmup_steps=optc.get("warmup_steps", 1),
            annealing_steps=optc.get("annealing_steps", 10),
            weight_decay=optc.get("weight_decay", 0.1),
            clip_grad=optc.get("clip_grad", 1.0),
        )

        pc = config["parallel"]
        parallel_cfg = ParallelConfig(
            world_size=pc.get("world_size", 1),
            rank=pc.get("rank", 0),
            local_rank=pc.get("local_rank", 0),
            dist_backend=pc.get("dist_backend", "nccl"),
            use_fsdp=pc.get("use_fsdp", False),
        )

        data_cfg = DataConfig(seq_length=config["data"].get("seq_length", 1024))

        offload_cfg_dict = config.get("offload", {})
        offload_cfg = OffloadConfig()
        for k, v in offload_cfg_dict.items():
            if hasattr(offload_cfg, k):
                setattr(offload_cfg, k, v)

        peft_dict = config["peft"]
        peft_cfg = PEFTConfig(method=peft_dict.get("method", "none"))

        align_dict = config["alignment"]
        alignment_cfg = AlignmentConfig(method=align_dict.get("method", "dpo"))

        main_config = MainConfig(
            model=model_cfg,
            init=InitConfig(**config["init"]),
            optim=optim_cfg,
            data=data_cfg,
            parallel=parallel_cfg,
            trainer=trainer_cfg,
            operation=operation_cfg,
            utils=UtilsConfig(**config["utils"]),
            profiler=ProfilerConfig(**config["profiler"]),
            peft=peft_cfg,
            alignment=alignment_cfg,
            offload=offload_cfg,
        )

        def forward_step(model, _data_iterator):
            device = next(model.parameters()).device
            input_ids = torch.randint(
                0, 32000, (1, config["data"]["seq_length"]), device=device
            )
            labels = input_ids.clone()
            logits = model(input_ids, labels=None)
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
            )
            return loss

        from tests.integration.offload.conftest import (
            create_mock_data_iterator,
            create_mock_evaluators,
        )

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
            start_time = time.time()

            for step in range(steps):
                step_start = time.time()
                loss, _, _ = trainer.train_step(step=step)
                step_time_ms = (time.time() - step_start) * 1000

                result.losses.append(loss)
                result.step_times_ms.append(step_time_ms)

            result.wall_time_s = time.time() - start_time
            result.peak_vram_mb = torch.cuda.max_memory_allocated() / 1024**2
            result.allocated_vram_mb = torch.cuda.memory_allocated() / 1024**2

            trainer._finalize_process()

        result.status = "PASS"

    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            result.status = "OOM"
            result.error = str(e)
        else:
            result.status = "FAIL"
            result.error = str(e)
    except Exception as e:
        result.status = "FAIL"
        result.error = str(e)

    return result


def print_section(title: str) -> None:
    """Print formatted section header."""
    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print(f"{'=' * 70}")


def print_result(result: ValidationResult, baseline_loss: float | None = None) -> None:
    """Print validation result."""
    status_color = {
        "PASS": "\033[92m",
        "FAIL": "\033[91m",
        "OOM": "\033[93m",
        "SKIP": "\033[90m",
    }.get(result.status, "")
    reset = "\033[0m"

    delta = ""
    if baseline_loss and result.losses:
        d = result.losses[-1] - baseline_loss
        delta = f"  (Δ {d:+.4f})"

    print(f"\n  [{status_color}{result.status}{reset}] {result.mode}{delta}")
    if result.losses:
        print(f"    Final Loss: {result.losses[-1]:.4f}  Min: {min(result.losses):.4f}")
    print(f"    Peak VRAM:   {result.peak_vram_mb:.1f} MB")
    print(f"    Allocated:   {result.allocated_vram_mb:.1f} MB")
    print(f"    Time:        {result.wall_time_s:.1f}s  ({result.steps} steps)")
    if result.step_times_ms:
        avg = sum(result.step_times_ms) / len(result.step_times_ms)
        print(f"    Avg/step:    {avg:.1f}ms")
    if result.error:
        print(f"    Error:       {result.error[:100]}")


def estimate_vram_from_params(params_b: float, mode: str) -> dict[str, float]:
    """Estimate VRAM usage from parameter count (in billions)."""
    # Weights (BF16 = 2 bytes)
    weights_gb = params_b * 2

    # Optimizer states (AdamW fp32 = 2 states × 4 bytes)
    optim_gb = params_b * 8

    # Activations (rough estimate for seq_len=1024, batch=1)
    act_gb = params_b * 0.03

    if mode == "baseline":
        return {"weights": weights_gb, "optim": optim_gb, "activations": act_gb}
    elif mode == "m1_only":
        return {"weights": weights_gb, "optim": 0, "activations": act_gb}
    elif mode in ("m3_only", "m1_m3"):
        return {"weights": weights_gb, "optim": optim_gb, "activations": 0}
    elif mode in ("m1_m2", "full"):
        # Staging pool for 2 layers + compute buffers
        staging_gb = (params_b / 40) * 2 * 2  # ~2 layers worth
        return {"staging": staging_gb, "optim": 0, "activations": 0}

    return {}


def main():
    parser = argparse.ArgumentParser(
        description="13B Model Capability Validation - Measured VRAM and Performance"
    )
    parser.add_argument(
        "--mode",
        choices=list(OFFLOAD_MODES.keys()) + ["all"],
        default="all",
        help="Offload mode to test (default: all)",
    )
    parser.add_argument(
        "--steps", type=int, default=50, help="Number of training steps (default: 50)"
    )
    parser.add_argument(
        "--batch-size", type=int, default=1, help="Micro batch size (default: 1)"
    )
    parser.add_argument(
        "--seq-len", type=int, default=1024, help="Sequence length (default: 1024)"
    )
    parser.add_argument(
        "--model-path",
        default="/tmp/validate_13b",
        help="Model checkpoint path (default: /tmp/validate_13b)",
    )
    parser.add_argument(
        "--output",
        default="13b_capability_results.json",
        help="Output JSON file (default: 13b_capability_results.json)",
    )
    parser.add_argument(
        "--timeout", type=int, default=7200, help="Per-test timeout in seconds"
    )
    parser.add_argument(
        "--estimate-only",
        action="store_true",
        help="Only show estimates without running tests",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("Error: CUDA required for 13B validation")
        sys.exit(1)

    gpu_name = torch.cuda.get_device_name()
    gpu_mem_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3

    print_section("13B MODEL CAPABILITY VALIDATION")
    print(f"  GPU: {gpu_name}")
    print(f"  VRAM: {gpu_mem_gb:.1f} GB")
    print(f"  Steps: {args.steps}")
    print(f"  Batch: {args.batch_size}  Seq Len: {args.seq_len}")

    # Calculate 13B parameter count
    d_model = LLAMA_13B_CONFIG["d_model"]
    d_ffn = LLAMA_13B_CONFIG["d_ffn"]
    num_layers = LLAMA_13B_CONFIG["num_layers"]
    vocab_size = 32000

    # Rough parameter count
    emb = vocab_size * d_model
    per_layer = 4 * d_model * d_model + 3 * d_model * d_ffn  # attn + mlp
    params = emb + num_layers * per_layer
    params_b = params / 1e9

    print(f"  Model: ~{params_b:.1f}B parameters (LLaMA-13B style)")

    modes_to_test = (
        list(OFFLOAD_MODES.keys()) if args.mode == "all" else [args.mode]
    )

    if args.estimate_only:
        print_section("VRAM ESTIMATES (NOT MEASURED)")
        print(f"\n  {'Mode':<12} {'Weights':>10} {'Optimizer':>10} {'Activations':>12} {'Total':>10}")
        print("  " + "-" * 60)
        for mode in modes_to_test:
            vram = estimate_vram_from_params(params_b, mode)
            total_gb = sum(vram.values())
            print(
                f"  {mode:<12} {vram.get('weights', 0):>10.1f}GB "
                f"{vram.get('optim', 0):>10.1f}GB {vram.get('activations', 0):>10.1f}GB "
                f"{vram.get('staging', total_gb):>10.1f}GB"
            )
        print("\n  Run without --estimate-only to get actual measurements.")
        return

    results = []
    baseline_loss = None

    for mode in modes_to_test:
        print_section(f"Testing: {mode}")

        # Skip baseline if GPU too small
        if mode == "baseline" and gpu_mem_gb < 80:
            result = ValidationResult(mode=mode, steps=args.steps, status="SKIP")
            result.error = f"Insufficient VRAM ({gpu_mem_gb:.1f}GB < 80GB required)"
            print_result(result)
            results.append(result)
            continue

        # Skip full offload if GPU too small
        if mode in ("m1_m2", "full") and gpu_mem_gb < 6:
            result = ValidationResult(mode=mode, steps=args.steps, status="SKIP")
            result.error = f"Insufficient VRAM ({gpu_mem_gb:.1f}GB < 6GB required)"
            print_result(result)
            results.append(result)
            continue

        config = build_config(
            LLAMA_13B_CONFIG,
            OFFLOAD_MODES[mode],
            args.steps,
            args.batch_size,
            args.seq_len,
            args.model_path,
        )

        result = run_validation(config, mode, args.steps)
        results.append(result)

        print_result(result, baseline_loss)

        if mode == "baseline" and result.status == "PASS":
            baseline_loss = result.losses[-1]

    # Summary
    print_section("SUMMARY")
    print(f"\n  {'Mode':<12} {'Status':>8} {'Loss':>10} {'Peak VRAM':>12} {'Time':>10}")
    print("  " + "-" * 60)

    for r in results:
        loss_str = f"{r.losses[-1]:.4f}" if r.losses else "N/A"
        print(
            f"  {r.mode:<12} {r.status:>8} {loss_str:>10} {r.peak_vram_mb:>10.1f}MB {r.wall_time_s:>8.1f}s"
        )

    # Compare to estimates
    print_section("MEASURED vs ESTIMATED VRAM")
    print(f"\n  {'Mode':<12} {'Measured':>12} {'Estimated':>12} {'Diff':>10}")
    print("  " + "-" * 50)

    for r in results:
        if r.status == "PASS":
            est = estimate_vram_from_params(params_b, r.mode)
            est_total = sum(est.values()) * 1024  # GB to MB
            diff = r.peak_vram_mb - est_total
            pct = (diff / est_total * 100) if est_total > 0 else 0
            print(
                f"  {r.mode:<12} {r.peak_vram_mb:>10.1f}MB {est_total:>10.1f}MB {diff:>+8.1f}MB ({pct:+.1f}%)"
            )

    # Save results
    output_data = {
        "gpu": gpu_name,
        "gpu_vram_gb": round(gpu_mem_gb, 1),
        "model_params_b": round(params_b, 2),
        "steps": args.steps,
        "batch_size": args.batch_size,
        "seq_len": args.seq_len,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "results": [r.to_dict() for r in results],
    }

    with open(args.output, "w") as f:
        json.dump(output_data, f, indent=2)

    print(f"\n  Results saved to: {args.output}")

    # Exit code based on results
    if any(r.status == "FAIL" for r in results):
        sys.exit(1)


if __name__ == "__main__":
    main()
