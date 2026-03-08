#!/usr/bin/env python3
"""Measure optimizer state memory specifically."""

import sys

import torch
import torch.distributed as dist

from ironcore.config import load_config
from ironcore.global_vars import set_global_states
from ironcore.model import LanguageModel
from ironcore.training_utils import get_loss_func


def get_optimizer_state_memory(optimizer):
    """Calculate total memory used by optimizer states in MiB."""
    total_bytes = 0
    for state in optimizer.state.values():
        if isinstance(state, dict):
            for v in state.values():
                if isinstance(v, torch.Tensor):
                    total_bytes += v.numel() * v.element_size()
    return total_bytes / 1024**2


def main():
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(rank)

    config_path = sys.argv[1]

    config = load_config(config_path)
    set_global_states(config)

    loss_fn = get_loss_func(config)
    model = LanguageModel(config, loss_fn).cuda()
    model = model.to(dtype=torch.bfloat16)

    total_params = sum(p.numel() for p in model.parameters())

    from ironcore.optimizer.muon import is_muon_param

    muon_param_count = sum(p.numel() for n, p in model.named_parameters() if is_muon_param(n, p))
    adamw_param_count = total_params - muon_param_count

    from ironcore.optimizer import get_optimizer

    optimizer = get_optimizer(config, model, "cuda")

    for p in model.parameters():
        if p.requires_grad:
            p.grad = torch.zeros_like(p)
    optimizer.step()

    opt_mem = get_optimizer_state_memory(optimizer)

    if rank == 0:
        optimizer_name = config.optim.optimizer.upper()
        print(f"\n{'=' * 50}")
        print(f"OPTIMIZER: {optimizer_name}")
        print(f"{'=' * 50}")
        print(f"Total model params: {total_params:,}")
        print(f"  Muon params: {muon_param_count:,} ({100 * muon_param_count / total_params:.1f}%)")
        print(
            f"  AdamW params: {adamw_param_count:,} ({100 * adamw_param_count / total_params:.1f}%)"
        )
        print("\nOptimizer state memory:")
        print(f"  Per rank: {opt_mem:.1f} MiB")
        print(f"  Total ({world_size} ranks): {opt_mem * world_size:.1f} MiB")
        print(f"{'=' * 50}\n")


if __name__ == "__main__":
    main()
