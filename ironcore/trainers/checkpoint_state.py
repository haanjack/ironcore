# Copyright (c) 2025-2026 Jaegeun Han
# SPDX-License-Identifier: Apache-2.0
"""Rank-local trainer state accompanying native model checkpoints."""

import random
from pathlib import Path

import numpy as np
import torch
from torch import distributed as dist
from torch.distributed.fsdp import (
    FullStateDictConfig,
    StateDictType,
)
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
)

from ironcore.parallel import parallel_states
from ironcore.parallel.random import (
    restore_tensor_parallel_rng_tracker,
    snapshot_tensor_parallel_rng_tracker,
)


def _path(trainer, step):
    rank = dist.get_rank() if dist.is_initialized() else 0
    return Path(trainer.config.trainer.model_path) / f"step_{step}" / f"trainer_rank{rank}.pt"


def save_trainer_state(trainer, step):
    """Preserve frozen references, AMP scale and each worker's RNG streams."""
    if trainer.config.operation.no_save or not trainer.config.trainer.model_path:
        return
    reference = getattr(trainer, "reference_model", None)
    reference_state = None
    if isinstance(reference, FSDP):
        with FSDP.state_dict_type(
            reference,
            StateDictType.FULL_STATE_DICT,
            FullStateDictConfig(offload_to_cpu=True, rank0_only=False),
        ):
            reference_state = reference.state_dict()
    elif reference is not None:
        reference_state = reference.state_dict()
    state = {
        "task": trainer.config.data.task_type,
        "world_size": dist.get_world_size() if dist.is_initialized() else 1,
        "tp_size": trainer.config.trainer.tensor_model_parallel_size,
        "cp_size": getattr(trainer.config.trainer, "context_parallel_size", 1),
        "scaler": trainer.scaler.state_dict(),
        "torch_cpu": torch.get_rng_state(),
        "python_rng": random.getstate(),
        "numpy_rng": {
            "name": np.random.get_state()[0],
            "keys": torch.from_numpy(np.random.get_state()[1].copy().astype(np.int64)),
            "position": np.random.get_state()[2],
            "has_gauss": np.random.get_state()[3],
            "cached_gaussian": np.random.get_state()[4],
        },
        "data": {
            split: iterator.state_dict()
            for split, iterator in trainer.data_iterator.items()
            if hasattr(iterator, "state_dict")
        },
        "parameter_precision": trainer.config.trainer.parameter_precision,
        "torch_cuda": torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
        "tp_rng": [
            {"key": list(key), "state": value.cpu()}
            for key, value in snapshot_tensor_parallel_rng_tracker().items()
        ],
        "reference": reference_state,
    }
    from ironcore.checkpointing.expert import atomic_save
    from ironcore.checkpointing.integrity import collective_checkpoint_action

    def write():
        path = _path(trainer, step)
        path.parent.mkdir(parents=True, exist_ok=True)
        atomic_save(state, path)

    collective_checkpoint_action(write)


def load_trainer_state(trainer, step):
    """Restore state after the policy and reference models have been created."""
    if step <= 0:
        return
    path = _path(trainer, step)
    if dist.is_initialized():
        available = torch.tensor(
            int(path.exists()),
            device=torch.device("cuda", torch.cuda.current_device())
            if dist.get_backend() == "nccl"
            else torch.device("cpu"),
        )
        dist.all_reduce(available)
        if 0 < available.item() < dist.get_world_size():
            raise RuntimeError(
                "Checkpoint is missing a rank-local trainer state; all ranks stopped"
            )
    if not path.exists():
        trainer.logger.warning(
            "Checkpoint has no rank-local trainer state; reference/AMP/RNG resume "
            "equivalence is unavailable (legacy checkpoint or changed topology)."
        )
        return
    error = None
    try:
        state = torch.load(path, map_location="cpu", weights_only=True)
        if state["task"] != trainer.config.data.task_type:
            raise ValueError(
                "Trainer resume requires the original task; initialize a new task with "
                "trainer.load_from_hf and a separate output model_path"
            )
        if (
            state["task"] == trainer.config.data.task_type
            and getattr(trainer, "reference_model", None) is not None
            and not state.get("reference")
        ):
            raise ValueError("Frozen reference state is empty")
    except Exception as caught:
        error = caught
    if dist.is_initialized():
        invalid = torch.tensor(
            int(error is not None),
            device=torch.device("cuda", torch.cuda.current_device())
            if dist.get_backend() == "nccl"
            else torch.device("cpu"),
        )
        dist.all_reduce(invalid, op=dist.ReduceOp.MAX)
        if invalid.item():
            raise RuntimeError(
                "Invalid rank-local trainer checkpoint; all ranks stopped"
            ) from error
    elif error is not None:
        raise error
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    if (state["world_size"], state["tp_size"], state.get("cp_size", 1)) != (
        world_size,
        trainer.config.trainer.tensor_model_parallel_size,
        getattr(trainer.config.trainer, "context_parallel_size", 1),
    ):
        raise ValueError("Trainer state resume requires the original DP/TP/CP topology")
    if (
        state.get("parameter_precision", trainer.config.trainer.parameter_precision)
        != trainer.config.trainer.parameter_precision
    ):
        raise ValueError("Exact trainer resume requires the original parameter_precision")
    if trainer.config.optim.load_checkpoint_optim_state:
        trainer.scaler.load_state_dict(state["scaler"])
    reference = getattr(trainer, "reference_model", None)
    if reference is not None and state["reference"] is not None:
        if isinstance(reference, FSDP):
            with FSDP.state_dict_type(
                reference,
                StateDictType.FULL_STATE_DICT,
                FullStateDictConfig(offload_to_cpu=True, rank0_only=False),
            ):
                reference.load_state_dict(state["reference"], strict=True)
        else:
            reference.load_state_dict(state["reference"], strict=True)
    torch.set_rng_state(state["torch_cpu"])
    if "python_rng" in state:
        random.setstate(state["python_rng"])
    if "numpy_rng" in state:
        rng = state["numpy_rng"]
        np.random.set_state(
            (
                rng["name"],
                rng["keys"].numpy().astype(np.uint32),
                rng["position"],
                rng["has_gauss"],
                rng["cached_gaussian"],
            )
        )
    for split, data_state in state.get("data", {}).items():
        trainer.data_iterator[split].load_state_dict(data_state)
    if torch.cuda.is_available() and state["torch_cuda"] is not None:
        torch.cuda.set_rng_state(state["torch_cuda"])
    restore_tensor_parallel_rng_tracker(
        {tuple(entry["key"]): entry["state"] for entry in state["tp_rng"]},
    )
    # Detect accidental restoration into a different initialized parallel mode.
    assert parallel_states.get_tensor_model_parallel_world_size() == state["tp_size"]
