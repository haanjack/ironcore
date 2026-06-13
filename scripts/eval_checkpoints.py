#!/usr/bin/env python3
"""Standalone checkpoint evaluation — load each checkpoint and compute validation loss.

Usage (single GPU):
    python scripts/eval_checkpoints.py --config configs/experiments/nanogpt_convergence_dp2.yaml

Usage (2-GPU DP):
    torchrun --nproc_per_node 2 scripts/eval_checkpoints.py \
        --config configs/experiments/nanogpt_convergence_dp2.yaml
"""

import argparse
import json
import math
import os
from contextlib import nullcontext
from pathlib import Path

import torch

from ironcore.dataloader import get_data_iterator
from ironcore.language_model import LanguageModel
from ironcore.parallel import initialize_parallelism, initialize_process
from ironcore.parallel.parallel_states import (
    get_data_parallel_group,
    get_data_parallel_world_size,
    initialize_model_parallel,
)
from ironcore.train import load_full_config
from ironcore.utils.device import get_device, get_model_dtype


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate all checkpoints in a directory")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint-dir", type=str, default=None)
    parser.add_argument(
        "--steps", type=str, default=None, help="Comma-separated steps (default: all)"
    )
    parser.add_argument("--num-batches", type=int, default=40)
    parser.add_argument("--eval-batch-size", type=int, default=12)
    return parser.parse_args()


def find_checkpoints(ckpt_dir: Path, steps: list[int] | None = None):
    result = []
    for d in sorted(ckpt_dir.iterdir()):
        if d.is_dir() and d.name.startswith("step_"):
            step = int(d.name.split("_")[1])
            if steps is None or step in steps:
                result.append((step, d))
    return result


@torch.no_grad()
def evaluate_model(model, data_iterator, num_batches, autocast_ctx, device):
    total_loss = 0.0
    model.eval()
    for _ in range(num_batches):
        batch = next(data_iterator["eval"])
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        with autocast_ctx:
            loss = model(input_ids, labels)
        total_loss += loss.item()

    avg_loss = total_loss / num_batches
    if get_data_parallel_world_size() > 1:
        t = torch.tensor(avg_loss, device=device)
        torch.distributed.all_reduce(
            t, op=torch.distributed.ReduceOp.SUM, group=get_data_parallel_group()
        )
        avg_loss = t.item() / get_data_parallel_world_size()

    return avg_loss


def main():
    args = parse_args()
    # Set WORLD_SIZE=2 so the DP=2 config passes validation,
    # then override to single-GPU after loading
    os.environ["WORLD_SIZE"] = "2"
    config = load_full_config(args.config)

    # Single-GPU eval overrides
    config.parallel.world_size = 1
    config.parallel.rank = 0
    config.parallel.local_rank = 0
    config.trainer.micro_batch_size = args.eval_batch_size
    config.trainer.train_batch_size = args.eval_batch_size
    config.trainer.gradient_accumulation_steps = 1
    config.operation.train_steps = 1

    ckpt_dir = Path(args.checkpoint_dir or config.trainer.model_path)
    steps = [int(s) for s in args.steps.split(",")] if args.steps else None
    checkpoints = find_checkpoints(ckpt_dir, steps)

    if not checkpoints:
        print(f"No checkpoints found in {ckpt_dir}")
        return

    config.trainer.eval_batch_size = args.eval_batch_size

    from ironcore.global_vars import set_global_states

    # Disable wandb for eval
    config.utils.wandb_project = ""
    os.environ["WANDB_MODE"] = "disabled"

    set_global_states(config)
    initialize_process(config)
    initialize_model_parallel(
        config.trainer.tensor_model_parallel_size,
        timeout_in_minutes=10,
    )

    device = get_device()
    dtype = get_model_dtype(config)

    from ironcore.training_utils import get_loss_func

    loss_fn = get_loss_func(config.data.task_type)
    model = LanguageModel(config, loss_fn).to(device=device, dtype=dtype)
    model = initialize_parallelism(config, model)

    autocast_ctx = (
        torch.autocast(device_type=device, dtype=dtype) if device != "cpu" else nullcontext()
    )

    data_iterator = get_data_iterator(config)

    is_rank0 = torch.distributed.get_rank() == 0

    if is_rank0:
        print(f"\n{'=' * 70}")
        print(f"Checkpoint Evaluation: {ckpt_dir}")
        print(
            f"DP={get_data_parallel_world_size()} | eval_batches={args.num_batches} | batch_size={args.eval_batch_size}"
        )
        print(f"{'=' * 70}")
        print(f"{'Step':>6}  {'Val Loss':>10}  {'PPL':>10}")
        print(f"{'-' * 6}  {'-' * 10}  {'-' * 10}")

    from ironcore.checkpointing.native import load_checkpoint

    results = []
    for step, ckpt_path in checkpoints:
        # Set model_path to parent dir and pass step explicitly
        config.trainer.model_path = str(ckpt_dir)
        config.optim.load_checkpoint_optim_state = False
        config.optim.load_checkpoint_lr_scheduler = False
        load_checkpoint(config, model, optimizer=None, lr_scheduler=None, step=step)

        avg_loss = evaluate_model(model, data_iterator, args.num_batches, autocast_ctx, device)
        ppl = math.exp(avg_loss) if avg_loss < 50 else float("inf")

        if is_rank0:
            print(f"{step:>6}  {avg_loss:>10.4f}  {ppl:>10.4f}")
        results.append({"step": step, "val_loss": avg_loss, "ppl": ppl})

    if is_rank0:
        print(f"{'=' * 70}\n")
        output_file = ckpt_dir / "eval_results.json"
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {output_file}")


if __name__ == "__main__":
    main()
