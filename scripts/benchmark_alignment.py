# Copyright (c) 2025-2026 Jaegeun Han
# SPDX-License-Identifier: Apache-2.0
"""Bounded real-data alignment pilots; launch with torchrun (DP=2).

Downloads are explicit inputs. No datasets or trained weights are committed.
Example: python -m torch.distributed.run --standalone --nproc_per_node=2 \
 scripts/benchmark_alignment.py --task sft --model-dir /tmp/ironcore-smollm135 \
 --gsm8k /tmp/ironcore-gsm8k --preferences /tmp/ironcore-ultrafeedback \
 --output /tmp/ironcore-alignment-sft
"""

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import torch
import torch.distributed as dist

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def encode_chat_response(tokenizer, messages, context):
    """Mask by character offsets; BPE can merge across a template boundary."""
    prefix = tokenizer.apply_chat_template(
        messages[:-1], tokenize=False, add_generation_prompt=True
    )
    text = tokenizer.apply_chat_template(messages, tokenize=False)
    if not text.startswith(prefix):
        raise ValueError("Chat template prompt/response text prefix mismatch")
    encoded = tokenizer(
        text,
        add_special_tokens=False,
        return_offsets_mapping=True,
        truncation=True,
        max_length=context + 1,
    )
    ids = torch.tensor(encoded["input_ids"])
    labels = ids[1:].clone()
    for index, (_, end) in enumerate(encoded["offset_mapping"][1:]):
        if end <= len(prefix):
            labels[index] = -100
    return ids[:-1], labels


def prepare(args, tokenizer):
    import numpy as np
    import pyarrow.parquet as pq

    source = args.preferences if args.task == "dpo" else args.gsm8k
    files = {split: next(source.rglob(f"{split}-*.parquet")) for split in ("train", "test")}
    samples = {}
    manifest = {"task": args.task, "seed": args.seed, "context": args.context, "splits": {}}
    for split, file in files.items():
        rows = pq.read_table(file).to_pylist()
        indices = np.random.default_rng(args.seed).permutation(len(rows))
        selected = []
        for index in indices:
            row = rows[int(index)]
            if args.task == "dpo":
                pair = {}
                for side in ("chosen", "rejected"):
                    messages = row[side]
                    inputs, labels = encode_chat_response(tokenizer, messages, args.context)
                    pair[f"{side}_input_ids"] = inputs
                    pair[f"{side}_labels"] = labels
                if any(
                    not (pair[f"{side}_labels"] != -100).any() for side in ("chosen", "rejected")
                ):
                    continue
                selected.append((int(index), pair))
            else:
                prompt = row["question"] + "\nGive the final numeric answer after ####."
                prompt_text = tokenizer.apply_chat_template(
                    [{"role": "user", "content": prompt}],
                    tokenize=False,
                    add_generation_prompt=True,
                )
                prefix = tokenizer.encode(prompt_text, add_special_tokens=False)
                if args.task == "grpo":
                    if len(prefix) + args.max_new_tokens > args.context:
                        continue
                    sample = {
                        "input_ids": torch.tensor(prefix),
                        "prompt": prompt,
                        "metadata": {"answer": row["answer"], "sample_id": int(index)},
                    }
                else:
                    inputs, labels = encode_chat_response(
                        tokenizer,
                        [
                            {"role": "user", "content": prompt},
                            {"role": "assistant", "content": row["answer"]},
                        ],
                        args.context,
                    )
                    if not (labels != -100).any():
                        continue
                    sample = {"input_ids": inputs, "labels": labels}
                selected.append((int(index), sample))
            if len(selected) >= (args.train_samples if split == "train" else args.eval_samples):
                break
        samples[split] = [s for _, s in selected]
        manifest["splits"][split] = {
            "file": str(file),
            "sha256": hashlib.sha256(file.read_bytes()).hexdigest(),
            "total_rows": len(rows),
            "selected_rows": [i for i, _ in selected],
        }
    return samples, manifest


class PilotIterator:
    def __init__(self, samples, batch_size, rank, world, task, pad, signature):
        self.samples = samples
        self.batch_size = batch_size
        self.rank = rank
        self.world = world
        self.task = task
        self.pad = pad
        self.signature = signature
        self.cursor = 0

    def __iter__(self):
        return self

    def state_dict(self):
        return {
            "cursor": self.cursor,
            "signature": self.signature,
            "rank": self.rank,
            "world": self.world,
            "batch_size": self.batch_size,
        }

    def load_state_dict(self, state):
        for key in ("signature", "rank", "world", "batch_size"):
            if state[key] != getattr(self, key):
                raise ValueError("Pilot data resume configuration mismatch")
        self.cursor = state["cursor"]

    def __next__(self):
        from torch.nn.utils.rnn import pad_sequence

        batch = []
        for _ in range(self.batch_size):
            batch.append(self.samples[(self.cursor * self.world + self.rank) % len(self.samples)])
            self.cursor += 1
        if self.task == "grpo":
            ids = pad_sequence(
                [s["input_ids"] for s in batch],
                batch_first=True,
                padding_value=self.pad,
                padding_side="left",
            )
            mask = pad_sequence(
                [torch.ones_like(s["input_ids"]) for s in batch],
                batch_first=True,
                padding_value=0,
                padding_side="left",
            )
            return {
                "input_ids": ids,
                "attention_mask": mask,
                "prompts": [s["prompt"] for s in batch],
                "metadata": [s["metadata"] for s in batch],
            }
        width = max(v.numel() for s in batch for v in s.values())
        return {
            key: torch.stack(
                [
                    torch.nn.functional.pad(
                        s[key],
                        (0, width - s[key].numel()),
                        value=-100 if key.endswith("labels") else self.pad,
                    )
                    for s in batch
                ]
            )
            for key in batch[0]
        }


def config_for(args):
    from ironcore.config import (
        AlignmentConfig,
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
    from ironcore.config.config_alignment import (
        GenerationConfig,
        RewardFunctionEntry,
        RewardManagerConfig,
    )
    from ironcore.config.config_model import BiasConfig, KVCacheConfig, PositionalEmbeddingConfig

    world = int(os.environ["WORLD_SIZE"])
    return MainConfig(
        model=ModelConfig(
            d_model=576,
            d_ffn=1536,
            num_layers=30,
            num_attention_heads=9,
            num_attention_groups=3,
            head_dim=64,
            max_seq_len=args.context,
            max_position_embeddings=8192,
            precision="bfloat16",
            activation_type="swiglu",
            ln_type="rmsnorm",
            ln_eps=1e-5,
            positional_embedding=PositionalEmbeddingConfig(type="rope", base=100000),
            bias=BiasConfig.none(),
            untie_embed=False,
            reset_attention_mask=False,
            reset_position_ids=False,
            dropout_embd=0,
            dropout_attn=0,
            dropout_mlp=0,
            tokenizer_type="sentencepiece",
            vocab_name_or_path=str(args.model_dir),
            hf_model_type="llama",
            kv_cache=KVCacheConfig(enabled=False),
        ),
        trainer=TrainerConfig(
            load_from_hf=str(args.model_dir),
            parameter_precision="float32",
            micro_batch_size=args.micro_batch,
            train_batch_size=args.global_batch,
            gradient_accumulation_steps=args.global_batch // (world * args.micro_batch),
            use_flash_attn=False,
            eval_batch_size=args.micro_batch,
            log_interval=1,
            recompute_linear_ce=args.task == "sft",
            loss_chunk_size=128,
            save_checkpoint_steps=args.steps,
            model_path=str(args.output / "checkpoint"),
        ),
        init=InitConfig(seed=args.seed),
        optim=OptimConfig(
            max_lr=args.lr,
            min_lr=args.lr,
            warmup_steps=0,
            annealing_steps=args.steps,
            clip_grad=1,
            weight_decay=0.01,
        ),
        data=DataConfig(task_type=args.task),
        parallel=ParallelConfig(
            rank=int(os.environ["RANK"]), local_rank=int(os.environ["LOCAL_RANK"]), world_size=world
        ),
        operation=OperationConfig(
            train_steps=args.steps, no_save=True, eval_samples=args.eval_samples
        ),
        utils=UtilsConfig(report_memory_usage=False),
        profiler=ProfilerConfig(),
        peft=PEFTConfig(),
        alignment=AlignmentConfig(
            method="grpo" if args.task == "grpo" else "dpo",
            dpo_beta=0.1,
            grpo_objective=args.objective,
            grpo_group_size=4,
            grpo_rollout_micro_group_size=2,
            grpo_num_epochs=2,
            grpo_beta=0.01,
            metrics_interval=1,
            generation=GenerationConfig(
                max_new_tokens=args.max_new_tokens,
                temperature=0.8,
                top_p=1.0,
                use_chat_template=False,
            ),
            reward_manager=RewardManagerConfig(
                functions=[RewardFunctionEntry(type="math")], num_workers=2
            ),
        ),
    )


def main(args):
    from transformers import AutoTokenizer

    from ironcore.trainers import DPOTrainer, GRPOTrainer, LanguageModelTrainer
    from ironcore.training_utils import batch_objective_count, forward_step, get_loss_func

    torch.set_num_threads(4)
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    torch.backends.cuda.matmul.allow_tf32 = False
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    samples, manifest = prepare(args, tokenizer)
    signature = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    cfg = config_for(args)
    cls = {"sft": LanguageModelTrainer, "dpo": DPOTrainer, "grpo": GRPOTrainer}[args.task]

    class PilotTrainer(cls):
        def _get_data_iterator(self):
            return {
                split: PilotIterator(
                    samples[source],
                    self.config.trainer.train_batch_size // cfg.parallel.world_size
                    if args.task == "grpo" and split == "train"
                    else args.micro_batch,
                    cfg.parallel.rank,
                    cfg.parallel.world_size,
                    args.task,
                    tokenizer.pad_token_id,
                    signature,
                )
                for split, source in [("train", "train"), ("eval", "test")]
            }

        def log_training(self, step, loss, *rest):
            self.losses.append(float(loss))
            super().log_training(step, loss, *rest)

    args.output.mkdir(parents=True, exist_ok=True)
    with PilotTrainer(cfg, forward_step, get_loss_func(args.task)) as trainer:
        trainer.losses = []
        trainer._pre_train_setup()

        def evaluate():
            trainer.data_iterator["eval"].cursor = 0
            if args.task == "grpo":
                with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
                    torch.manual_seed(args.seed + 100000 + cfg.parallel.rank)
                    return trainer.evaluate(0)
            trainer.model.eval()
            totals = torch.zeros(3, device="cuda", dtype=torch.float64)
            for _ in range(args.eval_samples // (args.micro_batch * cfg.parallel.world_size)):
                batch = next(trainer.data_iterator["eval"])
                loss, accuracy = trainer._eval_step(batch)
                units = batch_objective_count(batch, args.task)
                totals += totals.new_tensor([loss * units, accuracy * units, units])
            dist.all_reduce(totals)
            return {
                "loss": (totals[0] / totals[2]).item(),
                "accuracy": (totals[1] / totals[2]).item(),
            }

        before = evaluate()
        ref_before = (
            [p.detach().clone() for p in trainer.reference_model.parameters()]
            if args.task != "sft"
            else []
        )
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        start = time.perf_counter()
        trainer.train()
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        after = evaluate()
        frozen = (
            all(
                torch.equal(a, b) and b.grad is None and not b.requires_grad
                for a, b in zip(ref_before, trainer.reference_model.parameters(), strict=True)
            )
            if ref_before
            else None
        )
        result = {
            "task": args.task,
            "objective": args.objective if args.task == "grpo" else None,
            "before": before,
            "after": after,
            "losses": trainer.losses,
            "steps": args.steps,
            "eval_sampling_seed_per_rank": args.seed + 100000 if args.task == "grpo" else None,
            "parameters": sum(p.numel() for p in trainer.model.parameters()),
            "elapsed_seconds": elapsed,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "frozen_reference_exact": frozen,
            "training_manifest": manifest,
            "config": vars(args)
            | {
                "model_dir": str(args.model_dir),
                "gsm8k": str(args.gsm8k),
                "preferences": str(args.preferences),
                "output": str(args.output),
            },
            "status": "completed",
        }
        if dist.get_rank() == 0:
            (args.output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
        if frozen is False:
            raise AssertionError("Frozen reference changed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=["sft", "dpo", "grpo"], required=True)
    for name in ["model-dir", "gsm8k", "preferences", "output"]:
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--steps", type=int, default=24)
    parser.add_argument("--context", type=int, default=512)
    parser.add_argument("--micro-batch", type=int, default=1)
    parser.add_argument("--global-batch", type=int, default=4)
    parser.add_argument("--train-samples", type=int, default=128)
    parser.add_argument("--eval-samples", type=int, default=32)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=73)
    parser.add_argument("--objective", choices=["grpo", "gspo"], default="grpo")
    main(parser.parse_args())
