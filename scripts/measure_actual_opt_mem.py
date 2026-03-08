#!/usr/bin/env python3
"""Measure actual optimizer state memory - simplified version."""

import sys

import torch
import torch.distributed as dist
import yaml


def main():
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(rank)

    # Add path for imports
    sys.path.insert(0, "/app")

    from dataclasses import fields, is_dataclass

    from ironcore.config import MainConfig
    from ironcore.global_vars import set_global_states
    from ironcore.language_model import LanguageModel
    from ironcore.training_utils import get_loss_func

    config_path = sys.argv[1]

    # Load YAML and convert to config
    with open(config_path) as f:
        cfg_dict = yaml.safe_load(f)

    def dict_to_dataclass(d, cls):
        """Convert dict to dataclass instance."""
        if not is_dataclass(cls):
            return d
        kwargs = {}
        for field in fields(cls):
            if field.name in d:
                val = d[field.name]
                if is_dataclass(field.type) and isinstance(val, dict):
                    kwargs[field.name] = dict_to_dataclass(val, field.type)
                else:
                    kwargs[field.name] = val
            elif field.default != field.default_factory:
                kwargs[field.name] = field.default
        return cls(**kwargs)

    config = dict_to_dataclass(cfg_dict, MainConfig)
    set_global_states(config)

    loss_fn = get_loss_func(config)
    model = LanguageModel(config, loss_fn).cuda().to(dtype=torch.bfloat16)

    from ironcore.optimizer import get_optimizer

    optimizer = get_optimizer(config, model, "cuda")

    # Initialize states
    for p in model.parameters():
        if p.requires_grad:
            p.grad = torch.zeros_like(p, dtype=torch.float32)
    optimizer.step()

    # Measure actual optimizer state memory
    state_mem = 0
    state_count = 0
    for state in optimizer.state.values():
        if isinstance(state, dict):
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state_mem += v.numel() * v.element_size()
                    state_count += 1

    state_mem_mb = state_mem / 1024**2

    if rank == 0:
        opt_name = config.optim.optimizer.upper()
        total_params = sum(p.numel() for p in model.parameters())
        print(f"\n{'=' * 50}")
        print(f"OPTIMIZER: {opt_name}")
        print(f"{'=' * 50}")
        print(f"Total model params: {total_params:,}")
        print(f"Number of state tensors: {state_count}")
        print(f"Actual optimizer state memory (per rank): {state_mem_mb:.1f} MiB")
        print(f"Total across {world_size} ranks: {state_mem_mb * world_size:.1f} MiB")
        print(f"{'=' * 50}\n")


if __name__ == "__main__":
    main()
