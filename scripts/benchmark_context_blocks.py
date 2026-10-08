# Copyright (c) 2025-2026 Jaegeun Han
# SPDX-License-Identifier: Apache-2.0

"""Measure long-context MLP checkpoint granularity in one fresh CUDA torchrun job."""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import statistics
import time
from dataclasses import asdict
from pathlib import Path
from types import TracebackType

import torch
import torch.distributed as dist

from ironcore.config import _config_validation
from ironcore.layers.moe import MoEMLP
from ironcore.train import load_full_config
from ironcore.trainers import LanguageModelTrainer
from ironcore.training_utils import forward_step, loss_func
from ironcore.utils.memory import get_detailed_memory_breakdown
from ironcore.utils.mfu import MFUCalculator


def benchmark(args: argparse.Namespace) -> None:
    """Compare identical decoders with absent, full-size or bounded MLP checkpoints."""
    if not torch.cuda.is_available():
        raise RuntimeError("Context/block benchmark requires CUDA")
    if args.report.exists():
        raise ValueError("Use a fresh report path per measurement")
    config = load_full_config(str(args.config))
    if config.trainer.context_parallel_size < 2:
        raise ValueError("This benchmark expects context parallelism")
    cp = config.trainer.context_parallel_size
    config.model.max_seq_len = args.context
    config.model.max_position_embeddings = args.context
    config.data.seq_length = args.context
    config.model.moe.use_moe = args.model_type == "moe"
    config.model.moe.num_routed_experts = args.experts
    config.model.moe.expert_backend = "batched"
    config.model.moe.router_bias = True
    if args.tokenizer is not None:
        config.model.vocab_name_or_path = str(args.tokenizer)
        config.data.vocab_name_or_path = str(args.tokenizer)
    local_tokens = (args.context + cp - 1) // cp * config.trainer.micro_batch_size
    config.trainer.mlp_chunk_size = {
        "none": None,
        "full": local_tokens,
        "blocked": args.block_size,
    }[args.checkpoint_mode]
    config.operation.activation_recompute = False
    config.operation.no_save = True
    config.operation.train_steps = args.steps
    config.trainer.log_interval = args.steps
    config.optim.annealing_steps = max(config.optim.annealing_steps, args.steps)
    _config_validation(config)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "model_type": args.model_type,
        "checkpoint_mode": args.checkpoint_mode,
        "sequence_length": args.context,
        "local_tokens": local_tokens,
        "mlp_chunk_size": config.trainer.mlp_chunk_size,
        "steps": args.steps,
        "warmup_steps_excluded": args.warmup,
        "gpu": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "config": asdict(config),
    }
    records = []

    class MeasuredTrainer(LanguageModelTrainer):
        phase = "initialize"

        def __exit__(
            self,
            exc_type: type[BaseException] | None,
            exc_value: BaseException | None,
            traceback: TracebackType | None,
        ) -> bool | None:
            if exc_type is None:
                return super().__exit__(exc_type, exc_value, traceback)
            # A failed rank cannot join the normal teardown barrier. These are
            # disposable torchrun workers; let the launcher terminate peers.
            return False

        def _forward_micro_batch(self, step: int) -> tuple[torch.Tensor, dict[str, float] | None]:
            self.phase = "forward"
            result = super()._forward_micro_batch(step)
            self.phase = "backward"
            return result

        def _compute_grad_and_param_norms(self, step: int) -> tuple[float, float]:
            self.phase = "gradient-sync/norm"
            return super()._compute_grad_and_param_norms(step)

        def _optimizer_step(self) -> None:
            self.phase = "optimizer"
            return super()._optimizer_step()

        def train_step(self, step: int) -> tuple[float, float, float]:
            torch.cuda.synchronize()
            started = time.perf_counter()
            result = super().train_step(step)
            torch.cuda.synchronize()
            if not all(math.isfinite(float(value)) for value in result):
                raise RuntimeError(f"Nonfinite metrics at update {step + 1}")
            records.append(
                {
                    "step": step + 1,
                    "loss": float(result[0]),
                    "grad_norm": float(result[1]),
                    "param_norm": float(result[2]),
                    "seconds": time.perf_counter() - started,
                    "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
                }
            )
            print(
                f"BENCH_UPDATE rank={dist.get_rank()} step={step + 1} seconds={records[-1]['seconds']:.3f}",
                flush=True,
            )
            return result

    trainer = MeasuredTrainer(config, forward_step_func=forward_step, loss_fn=loss_func)
    try:
        with trainer:
            torch.cuda.reset_peak_memory_stats()
            trainer.train()
            if len(records) != args.steps:
                raise RuntimeError("Trainer did not complete all requested updates")
            row = {
                "rank": dist.get_rank(),
                "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
                "peak_reserved_mib": torch.cuda.max_memory_reserved() / 2**20,
                "memory_breakdown_mib": get_detailed_memory_breakdown(
                    trainer.model, trainer.optimizer
                ),
                "parameters_local": sum(
                    parameter.numel() for parameter in trainer.model.parameters()
                ),
                "expert_counts_by_layer": [
                    module._expert_selection_counts.tolist()
                    for module in trainer.model.modules()
                    if isinstance(module, MoEMLP)
                ],
                "records": records,
            }
            rows = [None] * dist.get_world_size()
            dist.all_gather_object(rows, row)
            if dist.get_rank() == 0:
                durations = [
                    max(rank["records"][step]["seconds"] for rank in rows)
                    for step in range(args.warmup, args.steps)
                ]
                seconds = statistics.mean(durations)
                active = copy.deepcopy(config.model)
                if active.moe.use_moe:
                    active.d_ffn = (
                        active.moe.num_shared_experts + active.moe.num_experts_per_token
                    ) * active.moe.expert_intermediate_size
                calculator = MFUCalculator.from_config(active, config.model.padded_vocab_size)
                report = {
                    **metadata,
                    "status": "passed",
                    "mean_update_seconds": seconds,
                    "update_std_seconds": statistics.stdev(durations) if len(durations) > 1 else 0,
                    "global_tokens_per_second": config.trainer.train_batch_size
                    * args.context
                    / seconds,
                    "peak_allocated_mib": max(rank["peak_allocated_mib"] for rank in rows),
                    "estimated_model_tflops_per_gpu": calculator.compute_tflops(
                        config.trainer.train_batch_size,
                        args.context,
                        seconds,
                        dist.get_world_size(),
                    ),
                    "ranks": rows,
                }
                args.report.write_text(json.dumps(report, indent=2, default=str) + "\n")
                print(
                    json.dumps(
                        {
                            key: value
                            for key, value in report.items()
                            if key not in {"ranks", "config"}
                        }
                    )
                )
    except Exception as error:
        rank = int(os.environ.get("RANK", "0"))
        failure = {
            **metadata,
            "status": "oom" if isinstance(error, torch.OutOfMemoryError) else "error",
            "phase": trainer.phase,
            "rank": rank,
            "completed_updates": len(records),
            "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
            "exception": str(error),
            "records": records,
        }
        path = args.report.with_name(f"{args.report.stem}.rank{rank}.failure.json")
        path.write_text(json.dumps(failure, indent=2, default=str) + "\n")
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path)
    parser.add_argument("--model-type", choices=["dense", "moe"], required=True)
    parser.add_argument("--checkpoint-mode", choices=["none", "full", "blocked"], required=True)
    parser.add_argument("--context", type=int, required=True)
    parser.add_argument("--experts", type=int, default=4)
    parser.add_argument("--block-size", type=int, default=512)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    args = parser.parse_args()
    if (
        args.warmup < 0
        or args.steps <= args.warmup
        or min(args.context, args.block_size, args.experts) < 1
    ):
        parser.error("Require positive sizes and steps > warmup >= 0")
    benchmark(args)


if __name__ == "__main__":
    main()
