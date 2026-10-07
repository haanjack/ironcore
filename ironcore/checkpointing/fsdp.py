# Copyright (c) 2025-2026 Jaegeun Han
# SPDX-License-Identifier: Apache-2.0
"""FSDP full model and named optimizer states with atomic checkpoint commit."""

import hashlib
import os
from pathlib import Path

import torch
import torch.distributed as dist
from torch.distributed.fsdp import (
    FullOptimStateDictConfig,
    FullStateDictConfig,
    StateDictType,
)
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
)

from .expert import atomic_save


def save_fsdp_checkpoint(config, model, optimizer, scheduler, step):
    root = Path(config.trainer.model_path) / f"step_{step}"
    root.mkdir(parents=True, exist_ok=True)
    with FSDP.state_dict_type(
        model,
        StateDictType.FULL_STATE_DICT,
        FullStateDictConfig(offload_to_cpu=True, rank0_only=False),
        FullOptimStateDictConfig(offload_to_cpu=True, rank0_only=False),
    ):
        state = {
            "version": 1,
            "step": step,
            "model": model.state_dict(),
            "optimizer": FSDP.optim_state_dict(model, optimizer),
            "scheduler": scheduler.state_dict(),
        }
    from .integrity import collective_checkpoint_action

    def write_model():
        if dist.get_rank() == 0:
            atomic_save(state, root / "fsdp_full.pt")

    collective_checkpoint_action(write_model)
    dist.barrier()

    def commit():
        if dist.get_rank() == 0:
            files = [
                root / "fsdp_full.pt",
                *[root / f"trainer_rank{r}.pt" for r in range(dist.get_world_size())],
            ]
            atomic_save(
                {"files": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in files}},
                root / "fsdp_manifest.json",
                True,
            )
            latest = Path(config.trainer.model_path) / "latest_step.txt"
            temporary = latest.with_suffix(".tmp")
            with temporary.open("w") as handle:
                handle.write(str(step))
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, latest)

    collective_checkpoint_action(commit)
    dist.barrier()


def load_fsdp_checkpoint(config, model, optimizer, scheduler, step):
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

    verify_manifest(directory, "fsdp_manifest.json")
    state = torch.load(directory / "fsdp_full.pt", map_location="cpu", weights_only=True)
    with FSDP.state_dict_type(
        model,
        StateDictType.FULL_STATE_DICT,
        FullStateDictConfig(offload_to_cpu=True, rank0_only=False),
        FullOptimStateDictConfig(offload_to_cpu=True, rank0_only=False),
    ):
        model.load_state_dict(state["model"], strict=True)
        if optimizer is not None and config.optim.load_checkpoint_optim_state:
            optimizer.load_state_dict(
                FSDP.optim_state_dict_to_load(model, optimizer, state["optimizer"])
            )
    if scheduler is not None and config.optim.load_checkpoint_lr_scheduler:
        scheduler.load_state_dict(state["scheduler"])
    return state["step"]
