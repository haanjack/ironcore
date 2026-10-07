# Copyright (c) 2025-2026 Jaegeun Han
# SPDX-License-Identifier: Apache-2.0
"""Real-text learning and synchronized 1/2-GPU training measurements.

Prepare a bounded TinyStories sample (downloads only requested byte budgets):
  python scripts/benchmark_training.py --prepare --data-dir /tmp/ironcore-corpus
Run using torchrun, with independent output directories for every configuration:
  torchrun --standalone --nproc_per_node=2 scripts/benchmark_training.py \
    --data-dir /tmp/ironcore-corpus --output /tmp/ironcore-benchmark \
    --model-size 50m --context 1024 --micro-batch 4 --global-batch 32

Corpus quality measurements and synthetic throughput runs are explicitly labelled.
"""

import argparse
import hashlib
import json
import os
import statistics
import sys
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def prepare(args):
    import requests
    import tiktoken
    from transformers import GPT2Tokenizer

    args.data_dir.mkdir(parents=True, exist_ok=True)
    GPT2Tokenizer.from_pretrained("gpt2").save_pretrained(args.data_dir / "tokenizer")
    encoding = tiktoken.get_encoding("gpt2")
    manifest = {"tokenizer": "gpt2", "splits": {}}
    for split, limit in [("train", args.train_bytes), ("valid", args.valid_bytes)]:
        filename = f"TinyStoriesV2-GPT4-{split}.txt"
        url = f"https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/{filename}"
        with requests.get(url, stream=True, timeout=60) as response:
            response.raise_for_status()
            content = bytearray()
            for chunk in response.iter_content(65536):
                content.extend(chunk)
                if len(content) >= limit:
                    break
            raw = bytes(content[:limit])
            # Preserve complete story boundaries and valid UTF-8.
            boundary = raw.rfind(b"<|endoftext|>")
            if boundary < 0:
                raise ValueError("Sample is too small for one complete story")
            raw = raw[: boundary + len(b"<|endoftext|>")]
            text = raw.decode("utf-8")
            tokens = np.array(
                encoding.encode(text, allowed_special={"<|endoftext|>"}), dtype=np.uint16
            )
            path = args.data_dir / f"{split}.bin"
            tokens.tofile(path)
            manifest["splits"][split] = {
                "url": url,
                "revision": response.headers.get("x-repo-commit"),
                "bytes": len(raw),
                "tokens": tokens.size,
                "text_sha256": hashlib.sha256(raw).hexdigest(),
                "token_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
    (args.data_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


def config_for(args):
    from ironcore.config import (
        DataConfig,
        InitConfig,
        MainConfig,
        ModelConfig,
        OperationConfig,
        OptimConfig,
        ParallelConfig,
        PEFTConfig,
        ProfilerConfig,
        TrainerConfig,
        UtilsConfig,
    )
    from ironcore.config.config_model import BiasConfig, PositionalEmbeddingConfig

    width, layers, ffn = {"50m": (384, 8, 1024), "130m": (640, 13, 1792)}[args.model_size]
    world = int(os.environ["WORLD_SIZE"])
    dp = world // args.tp
    if args.global_batch % (args.micro_batch * dp):
        raise ValueError("Global batch must be divisible by micro-batch times DP size")
    config = MainConfig(
        model=ModelConfig(
            d_model=width,
            d_ffn=ffn,
            num_layers=layers,
            max_seq_len=args.context,
            num_attention_heads=width // 64,
            num_attention_groups=width // 64,
            head_dim=64,
            dropout_attn=0,
            dropout_embd=0,
            dropout_mlp=0,
            precision=args.precision,
            vocab_name_or_path=str(args.data_dir / "tokenizer"),
            tokenizer_type="bbpe",
            ln_type="rmsnorm",
            positional_embedding=PositionalEmbeddingConfig(type="rope"),
            activation_type="swiglu",
            bias=BiasConfig.none(),
            untie_embed=True,
        ),
        trainer=TrainerConfig(
            micro_batch_size=args.micro_batch,
            train_batch_size=args.global_batch,
            gradient_accumulation_steps=args.global_batch // (args.micro_batch * dp),
            tensor_model_parallel_size=args.tp,
            parameter_precision=args.parameter_precision,
            loss_chunk_size=args.loss_chunk_size,
            recompute_linear_ce=args.recompute_linear_ce,
            use_flash_attn=args.attention == "flash",
            compile_model=args.compile,
            eval_batch_size=1,
            log_interval=10,
            model_path=str(args.output / "checkpoint"),
            save_checkpoint_steps=args.steps,
        ),
        init=InitConfig(seed=args.seed, init_std=0.02),
        optim=OptimConfig(
            max_lr=3e-4,
            min_lr=3e-5,
            warmup_steps=10,
            annealing_steps=max(1, args.steps - 10),
            clip_grad=1,
            weight_decay=0.1,
        ),
        data=DataConfig(
            task_type="pretrain",
            seq_length=args.context,
            vocab_name_or_path=str(args.data_dir / "tokenizer"),
        ),
        parallel=ParallelConfig(
            use_fsdp=args.fsdp,
            fsdp_use_orig_params=True,
            use_distributed_optimizer=args.distributed_optimizer,
            rank=int(os.environ["RANK"]),
            local_rank=int(os.environ["LOCAL_RANK"]),
            world_size=world,
        ),
        operation=OperationConfig(
            train_steps=args.steps,
            no_save=not args.save_checkpoint,
            activation_recompute=args.recompute,
        ),
        utils=UtilsConfig(report_memory_usage=False),
        profiler=ProfilerConfig(
            torch_profiler=args.profile,
            start=args.warmup + 1,
            end=args.warmup + 5,
            ranks=[int(rank) for rank in args.profile_ranks.split(",")],
            active_steps=2,
            output_dir=str(args.output / "profile"),
            export_chrome_trace=args.profile,
            export_csv=args.profile,
        ),
        peft=PEFTConfig(),
    )
    if args.moe:
        from ironcore.config.config_moe import MoEConfig

        config.model.moe = MoEConfig(
            use_moe=True,
            num_shared_experts=1,
            num_routed_experts=4,
            num_experts_per_token=2,
            expert_intermediate_size=args.expert_ffn or {"50m": 256, "130m": 384}[args.model_size],
            aux_loss_alpha=args.moe_aux_alpha,
            expert_backend=args.moe_backend,
            expert_model_parallel_size=args.ep,
            router_dtype="float32",
        )
    return config


def corpus_batches(args, config, split):
    from ironcore.parallel import parallel_states as ps

    tokens = np.memmap(args.data_dir / f"{split}.bin", dtype=np.uint16, mode="r")
    block_count = (len(tokens) - 1) // args.context
    if block_count < config.trainer.train_batch_size:
        raise ValueError("Corpus has too few blocks for one complete global batch")
    cursor = 0
    batch_size = config.trainer.micro_batch_size if split == "train" else 1
    dp_rank, dp_size = ps.get_data_parallel_group_rank(), ps.get_data_parallel_world_size()
    while True:
        blocks = []
        for _ in range(batch_size):
            index = (cursor * dp_size + dp_rank) % block_count
            cursor += 1
            start = index * args.context
            blocks.append(
                torch.from_numpy(tokens[start : start + args.context + 1].astype(np.int64))
            )
        batch = torch.stack(blocks)
        yield {"input_ids": batch[:, :-1].contiguous(), "labels": batch[:, 1:].contiguous()}


def benchmark(args):
    from ironcore.trainers import LanguageModelTrainer
    from ironcore.training_utils import forward_step, loss_func

    if not torch.cuda.is_available():
        raise RuntimeError("GPU benchmark requires CUDA; no CPU fallback")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    torch.set_num_threads(4)
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / "rank0.json").exists():
        raise ValueError("Use a new output directory for each measurement")
    if args.attention == "flash":
        from ironcore.layers.attention import flash_attn_varlen_func

        if flash_attn_varlen_func is None:
            raise RuntimeError("Requested FlashAttention extension is unavailable")
    cfg = config_for(args)

    class MeasuredTrainer(LanguageModelTrainer):
        def _phase(self, name):
            return torch.profiler.record_function(name) if args.profile else nullcontext()

        def _get_data_iterator(self):
            return {split: corpus_batches(args, self.config, split) for split in ("train", "valid")}

        def _forward_micro_batch(self, step):
            with self._phase("phase/forward"):
                result = super()._forward_micro_batch(step)
            if args.moe:
                from ironcore.training_utils import get_last_moe_aux_loss

                self.aux_sum += get_last_moe_aux_loss()
            return result

        def _compute_grad_and_param_norms(self, step):
            with self._phase("phase/gradient_norm_clip"):
                return super()._compute_grad_and_param_norms(step)

        def _optimizer_step(self):
            with self._phase("phase/optimizer"):
                return super()._optimizer_step()

        def train_step(self, step):
            self.aux_sum = 0.0
            torch.cuda.synchronize()
            started = time.perf_counter()
            with self._phase(f"training_update/{step + 1}"):
                result = super().train_step(step)
            torch.cuda.synchronize()
            elapsed = torch.tensor(
                time.perf_counter() - started, device="cuda", dtype=torch.float64
            )
            dist.all_reduce(elapsed, op=dist.ReduceOp.MAX)
            self.records.append(
                {
                    "step": step + 1,
                    "seconds_slowest_rank": elapsed.item(),
                    "loss": result[0],
                    "grad_norm": result[1],
                    "lr": self.optimizer.param_groups[0]["lr"],
                    "aux_loss_local": self.aux_sum
                    / self.config.trainer.gradient_accumulation_steps,
                }
            )
            return result

    with MeasuredTrainer(cfg, forward_step, loss_func) as trainer:
        trainer.records = []
        unwrapped = getattr(trainer.model, "module", trainer.model)
        if hasattr(unwrapped, "_orig_mod"):
            unwrapped = unwrapped._orig_mod
        parameters = sum(p.numel() for p in unwrapped.parameters())
        parameters_full = sum(
            p.numel() * (args.tp if getattr(p, "tp_shard_dim", None) is not None else 1)
            for p in unwrapped.parameters()
        )
        routed_parameters = sum(
            p.numel() * (args.tp if getattr(p, "tp_shard_dim", None) is not None else 1)
            for name, p in unwrapped.named_parameters()
            if ".routed_experts." in name
        )
        if args.ep > 1:
            parameters_full += routed_parameters * (args.ep - 1)
            routed_parameters *= args.ep
        if args.fsdp:
            from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

            with FSDP.summon_full_params(trainer.model, writeback=False):
                parameters_full = sum(p.numel() for p in trainer.model.parameters())
        active_parameters = (
            parameters_full - routed_parameters + routed_parameters * 2 // 4
            if args.moe
            else parameters_full
        )

        def expert_stats():
            from ironcore.layers.moe import MoEMLP

            stats = {}
            for name, module in unwrapped.named_modules():
                if isinstance(module, MoEMLP):
                    counts = module._expert_selection_counts.clone()
                    if dist.get_world_size() // args.tp > 1:
                        from ironcore.parallel.parallel_states import get_data_parallel_group

                        dist.all_reduce(counts, group=get_data_parallel_group())
                    values = counts.double()
                    fractions = values / values.sum().clamp(min=1)
                    stats[name] = {
                        "counts": counts.tolist(),
                        "fractions": fractions.tolist(),
                        "max_over_mean": (values.max() / values.mean().clamp(min=1)).item(),
                    }
            return stats

        def heldout_loss():
            trainer.model.eval()
            batches = corpus_batches(args, cfg, "valid")
            losses = [trainer._eval_step(next(batches))[0] for _ in range(args.eval_batches)]
            value = torch.tensor(statistics.mean(losses), device="cuda")
            from ironcore.parallel.parallel_states import (
                get_data_parallel_group,
                get_data_parallel_world_size,
            )

            dist.all_reduce(value, group=get_data_parallel_group())
            value /= get_data_parallel_world_size()
            trainer.model.train()
            return value.item()

        initial_eval = heldout_loss()
        torch.cuda.reset_peak_memory_stats()
        trainer.train()
        peak_allocated = torch.cuda.max_memory_allocated()
        peak_reserved = torch.cuda.max_memory_reserved()
        final_eval = heldout_loss()
        measured = trainer.records[args.warmup :]
        durations = [record["seconds_slowest_rank"] for record in measured]
        seconds = statistics.mean(durations)
        result = {
            "model_size_label": args.model_size,
            "parameters_local": parameters,
            "parameters_full": parameters_full,
            "active_parameters_per_token_estimate": active_parameters,
            "moe": args.moe,
            "moe_config": vars(cfg.model.moe) if args.moe else None,
            "expert_stats": expert_stats(),
            "world_size": dist.get_world_size(),
            "tp": args.tp,
            "ep": args.ep,
            "fsdp": args.fsdp,
            "distributed_optimizer": args.distributed_optimizer,
            "context": args.context,
            "micro_batch": args.micro_batch,
            "global_batch": args.global_batch,
            "precision": args.precision,
            "parameter_precision": args.parameter_precision,
            "attention": args.attention,
            "compile": args.compile,
            "activation_recompute": args.recompute,
            "loss_chunk_size": args.loss_chunk_size,
            "seed": args.seed,
            "torch": torch.__version__,
            "recompute_linear_ce": args.recompute_linear_ce,
            "moe_backend": args.moe_backend,
            "profile": args.profile,
            "profile_ranks": cfg.profiler.ranks,
            "profiler_schedule": {
                "start": cfg.profiler.start,
                "end": cfg.profiler.end,
                "wait": cfg.profiler.wait_steps,
                "warmup": cfg.profiler.warmup_steps,
                "active": cfg.profiler.active_steps,
            },
            "gpu": torch.cuda.get_device_name(),
            "steps": args.steps,
            "warmup_steps_excluded": args.warmup,
            "mean_step_seconds": seconds,
            "median_step_seconds": statistics.median(durations),
            "p95_step_seconds": sorted(durations)[
                min(len(durations) - 1, int(len(durations) * 0.95))
            ],
            "global_tokens_per_second": args.global_batch * args.context / seconds,
            "peak_allocated_bytes": peak_allocated,
            "peak_reserved_bytes": peak_reserved,
            "initial_heldout_loss": initial_eval,
            "final_heldout_loss": final_eval,
            "corpus_manifest": json.loads((args.data_dir / "manifest.json").read_text()),
            "records": trainer.records,
        }
        (args.output / f"rank{dist.get_rank()}.json").write_text(
            json.dumps(result, indent=2) + "\n"
        )
        if dist.get_rank() == 0:
            print(
                json.dumps(
                    {k: v for k, v in result.items() if k not in ("records", "corpus_manifest")},
                    indent=2,
                )
            )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--prepare", action="store_true")
    p.add_argument("--data-dir", type=Path, required=True)
    p.add_argument("--output", type=Path, default=Path("outputs/benchmark_training"))
    p.add_argument("--model-size", choices=["50m", "130m"], default="50m")
    p.add_argument("--context", type=int, default=1024)
    p.add_argument("--micro-batch", type=int, default=4)
    p.add_argument("--global-batch", type=int, default=32)
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--eval-batches", type=int, default=8)
    p.add_argument("--tp", type=int, default=1)
    p.add_argument("--ep", type=int, choices=[1, 2], default=1)
    p.add_argument("--fsdp", action="store_true")
    p.add_argument("--distributed-optimizer", action="store_true")
    p.add_argument("--precision", choices=["float32", "bfloat16"], default="bfloat16")
    p.add_argument("--parameter-precision", choices=["float32", "model"], default="float32")
    p.add_argument("--attention", choices=["sdpa", "flash"], default="sdpa")
    p.add_argument("--compile", action="store_true")
    p.add_argument("--recompute", action="store_true")
    p.add_argument("--save-checkpoint", action="store_true")
    p.add_argument("--profile", action="store_true")
    p.add_argument("--profile-ranks", default="0", help="Comma-separated global ranks to capture")
    p.add_argument("--loss-chunk-size", type=int)
    p.add_argument("--recompute-linear-ce", action="store_true")
    p.add_argument("--moe-backend", choices=["loop", "batched"], default="loop")
    p.add_argument("--moe", action="store_true", help="4 routed experts, top-2, 1 shared expert")
    p.add_argument("--expert-ffn", type=int)
    p.add_argument("--moe-aux-alpha", type=float, default=0.01)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--train-bytes", type=int, default=8 * 1024 * 1024)
    p.add_argument("--valid-bytes", type=int, default=1024 * 1024)
    args = p.parse_args()
    if (
        args.steps <= args.warmup
        or min(args.context, args.micro_batch, args.global_batch, args.eval_batches) < 1
    ):
        p.error("Positive sizes and steps > warmup are required")
    (prepare if args.prepare else benchmark)(args)


if __name__ == "__main__":
    main()
