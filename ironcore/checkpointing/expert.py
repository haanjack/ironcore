# Copyright (c) 2025-2026 Jaegeun Han
# SPDX-License-Identifier: Apache-2.0
"""Same-topology EP checkpoint with explicit global expert identity."""

import hashlib
import json
import os
import re
from pathlib import Path

import torch
import torch.distributed as dist


def global_parameter_names(model):
    model = getattr(model, "module", model)
    modules = dict(model.named_modules())
    result = {}
    for name in model.state_dict():
        match = re.search(r"(.*)routed_experts\.(\d+)\.(.*)", name)
        if match:
            prefix, index, suffix = match.groups()
            owner = modules[prefix.rstrip(".")]
            result[name] = f"{prefix}routed_experts.{owner.expert_start_idx + int(index)}.{suffix}"
        else:
            result[name] = name
    return result


def atomic_save(value, path, json_value=False):
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w" if json_value else "wb") as handle:
        if json_value:
            json.dump(value, handle, indent=2)
        else:
            torch.save(value, handle)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def save_expert_checkpoint(config, model, optimizer, scheduler, step):
    rank = dist.get_rank()
    root = Path(config.trainer.model_path) / f"step_{step}"
    root.mkdir(parents=True, exist_ok=True)
    module = getattr(model, "module", model)
    names = global_parameter_names(module)
    record = {
        "version": 1,
        "step": step,
        "rank": rank,
        "world_size": dist.get_world_size(),
        "ep_size": config.model.moe.expert_model_parallel_size,
        "tp_size": config.trainer.tensor_model_parallel_size,
        "model": {names[n]: t.cpu() for n, t in module.state_dict().items()},
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "optimizer_parameter_names": [
            [
                names[next(n for n, p in module.named_parameters() if p is param)]
                for param in g["params"]
            ]
            for g in optimizer.param_groups
        ],
    }
    from .integrity import collective_checkpoint_action

    collective_checkpoint_action(lambda: atomic_save(record, root / f"ep{rank}.pt"))
    dist.barrier()

    def commit():
        if rank == 0:
            files = [root / f"ep{r}.pt" for r in range(dist.get_world_size())]
            files += [root / f"trainer_rank{r}.pt" for r in range(dist.get_world_size())]
            manifest = {
                "version": 1,
                "step": step,
                "files": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
            }
            atomic_save(manifest, root / "ep_manifest.json", True)
            latest = Path(config.trainer.model_path) / "latest_step.txt"
            temporary = latest.with_suffix(".tmp")
            with temporary.open("w") as handle:
                handle.write(str(step))
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, latest)

    collective_checkpoint_action(commit)
    dist.barrier()


def load_expert_checkpoint(config, model, optimizer, scheduler, step):
    root = Path(config.trainer.model_path)
    if not root.exists():
        return -1
    if step < 0:
        latest = root / "latest_step.txt"
        if not latest.exists():
            return -1
        step = int(latest.read_text())
    directory = root / f"step_{step}"
    from .integrity import verify_manifest

    verify_manifest(directory, "ep_manifest.json")
    state = torch.load(directory / f"ep{dist.get_rank()}.pt", map_location="cpu", weights_only=True)
    if (state["world_size"], state["ep_size"], state["tp_size"]) != (
        dist.get_world_size(),
        config.model.moe.expert_model_parallel_size,
        config.trainer.tensor_model_parallel_size,
    ):
        raise ValueError("EP checkpoint requires original world/EP/TP topology")
    module = getattr(model, "module", model)
    names = global_parameter_names(module)
    module.load_state_dict({n: state["model"][g] for n, g in names.items()}, strict=True)
    if optimizer is not None and config.optim.load_checkpoint_optim_state:
        actual = [
            [
                names[next(n for n, p in module.named_parameters() if p is param)]
                for param in g["params"]
            ]
            for g in optimizer.param_groups
        ]
        if actual != state["optimizer_parameter_names"]:
            raise ValueError("EP optimizer global parameter identities differ")
        optimizer.load_state_dict(state["optimizer"])
    if scheduler is not None and config.optim.load_checkpoint_lr_scheduler:
        scheduler.load_state_dict(state["scheduler"])
    return state["step"]
