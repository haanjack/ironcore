# Copyright (c) 2025-2026 Jaegeun Han
# SPDX-License-Identifier: Apache-2.0

"""Measure one MoE scaling point through the production trainer under torchrun."""

from __future__ import annotations

import argparse
import copy
import json
import math
import statistics
import time
from pathlib import Path

import torch
import torch.distributed as dist

from ironcore.layers.moe import MoEMLP
from ironcore.train import load_full_config
from ironcore.trainers import LanguageModelTrainer
from ironcore.training_utils import forward_step, loss_func
from ironcore.utils.memory import get_detailed_memory_breakdown
from ironcore.utils.mfu import MFUCalculator


def benchmark(args: argparse.Namespace) -> None:
    """Run a fresh process per point; collect synchronized updates and rank peaks."""
    if not torch.cuda.is_available():
        raise RuntimeError("MoE scaling requires CUDA")
    if args.report.exists():
        raise ValueError("Use a new report path for every measurement")
    config = load_full_config(str(args.config))
    moe = config.model.moe
    if not moe.use_moe or moe.expert_model_parallel_size != 1:
        raise ValueError("This benchmark compares EP1 expert backends")
    if args.experts < moe.num_experts_per_token:
        raise ValueError("Expert count must be at least top-k")
    moe.num_routed_experts = args.experts
    moe.expert_backend = args.backend
    # Identical router architecture in natural and forced cases.
    moe.router_bias = True
    config.operation.train_steps = args.steps
    config.operation.no_save = True
    config.trainer.log_interval = args.steps
    config.optim.annealing_steps = max(config.optim.annealing_steps, args.steps)
    records = []

    class MeasuredTrainer(LanguageModelTrainer):
        def train_step(self, step: int) -> tuple[float, float, float]:
            torch.cuda.synchronize()
            started = time.perf_counter()
            result = super().train_step(step)
            torch.cuda.synchronize()
            seconds = time.perf_counter() - started
            if not all(math.isfinite(float(value)) for value in result):
                raise RuntimeError(f"Nonfinite training metrics at update {step + 1}")
            records.append(
                {
                    "step": step + 1,
                    "loss": float(result[0]),
                    "grad_norm": float(result[1]),
                    "seconds": seconds,
                }
            )
            return result

    with MeasuredTrainer(config, forward_step_func=forward_step, loss_fn=loss_func) as trainer:
        layers = [module for module in trainer.model.modules() if isinstance(module, MoEMLP)]
        if not layers:
            raise ValueError("No MoE layers found")
        if args.routing == "forced":
            with torch.no_grad():
                for layer in layers:
                    layer.router.weight.zero_()
                    layer.router.bias.fill_(-4)
                    for expert in range(moe.num_experts_per_token):
                        layer.router.bias[expert] = 4 - expert
        torch.cuda.reset_peak_memory_stats()
        trainer.train()
        if len(records) != args.steps:
            raise RuntimeError("Trainer did not complete all requested updates")
        row = {
            "rank": dist.get_rank(),
            "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
            "peak_reserved_mib": torch.cuda.max_memory_reserved() / 2**20,
            "memory_breakdown_mib": get_detailed_memory_breakdown(trainer.model, trainer.optimizer),
            "parameters_local": sum(parameter.numel() for parameter in trainer.model.parameters()),
            "expert_counts_by_layer": [layer._expert_selection_counts.tolist() for layer in layers],
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
            active.d_ffn = (
                moe.num_shared_experts + moe.num_experts_per_token
            ) * moe.expert_intermediate_size
            calculator = MFUCalculator.from_config(active, config.model.padded_vocab_size)
            total_tokens = config.trainer.train_batch_size * config.data.seq_length
            report = {
                "experts": args.experts,
                "backend": args.backend,
                "routing": args.routing,
                "steps": args.steps,
                "warmup_steps_excluded": args.warmup,
                "cp": config.trainer.context_parallel_size,
                "tp": config.trainer.tensor_model_parallel_size,
                "sequence_length": config.data.seq_length,
                "global_batch_size": config.trainer.train_batch_size,
                "top_k": moe.num_experts_per_token,
                "virtual_block_size": moe.virtual_block_size,
                "grouped_token_budget": moe.grouped_token_budget,
                "mlp_chunk_size": config.trainer.mlp_chunk_size,
                "mean_update_seconds": seconds,
                "update_std_seconds": statistics.stdev(durations) if len(durations) > 1 else 0,
                "global_tokens_per_second": total_tokens / seconds,
                "peak_allocated_mib": max(rank["peak_allocated_mib"] for rank in rows),
                "estimated_model_tflops_per_gpu": calculator.compute_tflops(
                    config.trainer.train_batch_size,
                    config.data.seq_length,
                    seconds,
                    dist.get_world_size(),
                ),
                "torch": torch.__version__,
                "gpu": torch.cuda.get_device_name(),
                "ranks": rows,
            }
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text(json.dumps(report, indent=2) + "\n")
            print(json.dumps({key: value for key, value in report.items() if key != "ranks"}))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--experts", type=int, required=True)
    parser.add_argument("--backend", choices=["batched", "grouped"], required=True)
    parser.add_argument("--routing", choices=["natural", "forced"], default="natural")
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=2)
    args = parser.parse_args()
    if args.warmup < 0 or args.steps <= args.warmup:
        parser.error("Require steps > warmup >= 0")
    benchmark(args)


if __name__ == "__main__":
    main()
