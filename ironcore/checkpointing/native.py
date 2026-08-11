# Copyright (c) 2025-2026 Jaegeun Han
#
# SPDX-License-Identifier: Apache-2.0

import dataclasses
import os
import pathlib
import random
from pathlib import Path
from typing import Union

import torch
from torch import distributed as dist
from torch.optim.lr_scheduler import LRScheduler

from ironcore.config import (
    DataConfig,
    LoRAConfig,
    MainConfig,
    PEFTConfig,
)
from ironcore.config.config_model import ModelConfig, PositionalEmbeddingConfig
from ironcore.config.config_optim import OptimConfig
from ironcore.config.config_parallel import ParallelConfig
from ironcore.config.config_trainer import InitConfig, OperationConfig, TrainerConfig
from ironcore.config.config_utils import UtilsConfig
from ironcore.global_vars import get_logger, get_timer
from ironcore.language_model import LanguageModel
from ironcore.optimizer import Optimizer
from ironcore.parallel import parallel_states
from ironcore.parallel.tensor_parallel import comm
from ironcore.utils import is_first_rank

_CKPT_FILENAME = "pytorch_model.bin"
_LATEST_STEP_FILENAME = "latest_step.txt"


class HFConfigManager:
    """Manage configuration files for HuggingFace compatibility."""

    @staticmethod
    def get_hf_config(config: MainConfig) -> dict:
        """Convert MainConfig to HuggingFace compatible config dict."""
        # Ensure model-specific HF names are provided for compatibility.
        model_type = config.model.hf_model_type
        architecture = config.model.hf_architecture

        if model_type is None or architecture is None:
            raise ValueError(
                "For HuggingFace compatibility, 'hf_model_type' and 'hf_architecture' "
                "must be set in the model config. Found at least one None value."
            )

        hf_config = {
            "model_type": model_type,
            "hidden_size": config.model.d_model,
            "num_hidden_layers": config.model.num_layers,
            "num_attention_heads": config.model.num_attention_heads,
            "intermediate_size": config.model.d_ffn,
            "max_position_embeddings": config.model.max_position_embeddings,
            "vocab_size": config.data.vocab_size,
            "layer_norm_eps": config.model.ln_eps,
            "initializer_range": config.init.init_std,
            "hidden_act": config.model.activation_type,
            "architectures": [architecture],
        }
        return hf_config

    @staticmethod
    def save_hf_config(config: MainConfig, save_directory: Union[str, Path]):
        """Save HuggingFace compatible config file."""
        import json

        hf_config = HFConfigManager.get_hf_config(config)
        if isinstance(save_directory, str):
            save_directory = Path(save_directory)
        hf_config_path = save_directory / "config.json"

        with open(hf_config_path, "w", encoding="utf-8") as f:
            json.dump(hf_config, f, indent=4)

    @staticmethod
    def load_hf_config(load_directory: Union[str, Path]) -> dict:
        """Load HuggingFace compatible config file."""
        import json

        if isinstance(load_directory, str):
            load_directory = Path(load_directory)
        hf_config_path = load_directory / "config.json"

        if not hf_config_path.exists():
            raise FileNotFoundError(f"HuggingFace config file {hf_config_path} does not exist.")

        with open(hf_config_path, encoding="utf-8") as f:
            hf_config = json.load(f)

        return hf_config


def _is_distributed_optimizer(optimizer):
    """Check if optimizer is a DistributedOptimizer wrapper."""
    return hasattr(optimizer, "optimizer") and hasattr(optimizer, "local_param_indices")


def _gather_distributed_optimizer_states(optimizer, model, dp_group):
    """Gather partitioned optimizer states from all DP ranks for universal checkpoint.

    Args:
        optimizer: DistributedOptimizer instance
        model: The model (for parameter mapping)
        dp_group: Data parallel process group

    Returns:
        dict: Full optimizer state dict with all parameters
    """
    dp_size = dist.get_world_size(group=dp_group)
    dp_rank = dist.get_rank(group=dp_group)

    # Build mapping once to avoid O(N^2) complexity
    param_to_name = {p: n for n, p in model.named_parameters()}

    # Get parameter list in same order as DistributedOptimizer
    all_params = []
    for group in optimizer.optimizer.param_groups:
        for p in group["params"]:
            all_params.append(p)

    # Each rank prepares its owned states
    local_owned_states = {}
    for param_idx, param in enumerate(all_params):
        if param_idx % dp_size == dp_rank:
            name = param_to_name.get(param)
            if name:
                local_state = optimizer.optimizer.state.get(param, {})
                # Serialize state for gather
                state_dict = {}
                for k, v in local_state.items():
                    if isinstance(v, torch.Tensor):
                        state_dict[k] = v.cpu()
                    else:
                        state_dict[k] = v
                local_owned_states[name] = state_dict

    # Gather all partial states at once
    full_state = {"state": {}, "param_groups": optimizer.optimizer.state_dict()["param_groups"]}

    if dp_size > 1:
        # Gather only to rank 0 to save memory on other ranks
        all_ranks_states = [None] * dp_size if dp_rank == 0 else None
        dist.gather_object(local_owned_states, all_ranks_states, dst=0, group=dp_group)

        if dp_rank == 0:
            # Merge into full state dict
            for rank_state in all_ranks_states:
                full_state["state"].update(rank_state)
    else:
        full_state["state"] = local_owned_states

    return full_state


def _partition_optimizer_states_for_load(optimizer, full_state_dict, model):
    """Partition full optimizer state dict for DistributedOptimizer.

    Takes a complete optimizer state dict and filters it to only include
    states for parameters owned by this DP rank.

    Args:
        optimizer: DistributedOptimizer instance
        full_state_dict: Complete optimizer state dict
        model: The model (for parameter mapping)

    Returns:
        dict: Partitioned optimizer state dict
    """
    # Build mapping once
    param_to_name = {p: n for n, p in model.named_parameters()}

    # Get parameter list in same order as DistributedOptimizer
    all_params = []
    for group in optimizer.optimizer.param_groups:
        for p in group["params"]:
            all_params.append(p)

    # Build partitioned state
    partitioned_state = {
        "state": {},
        "param_groups": full_state_dict["param_groups"],
    }

    for param_idx, param in enumerate(all_params):
        if param_idx not in optimizer.local_param_indices:
            continue  # Skip params not owned by this rank

        param_name = param_to_name.get(param)
        if param_name is None or param_name not in full_state_dict["state"]:
            continue

        # Copy state for this parameter
        state = full_state_dict["state"][param_name]
        partitioned_state["state"][param] = {}
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                partitioned_state["state"][param][k] = v.to(param.device)
            else:
                partitioned_state["state"][param][k] = v

    return partitioned_state


def load_checkpoint(
    config: MainConfig,
    model: torch.nn.Module,
    optimizer: Optimizer | None = None,
    lr_scheduler: LRScheduler | None = None,
    step: int = -1,
) -> int:
    """Load a checkpoint and restore the model and optimizer states."""

    logger = get_logger()
    timer = get_timer()

    if config.trainer.model_path == "":
        config.operation.no_save = True
        return -1

    if not Path(config.trainer.model_path).exists():
        return -1

    # determine trained step to load
    if step >= 0:
        logger.info(f"Loading checkpoint at step {step}")
    else:
        # find the latest checkpoint file
        latest_file = Path(config.trainer.model_path) / _LATEST_STEP_FILENAME
        if not latest_file.exists():
            logger.warning(f"Latest checkpoint file {latest_file} does not exist")
            return -1

        with open(
            Path(config.trainer.model_path) / _LATEST_STEP_FILENAME,
            encoding="utf-8",
        ) as f:
            step = int(f.read().strip())

    # checkpoint file name
    init_ckpt_path = Path(config.trainer.model_path) / f"step_{step}"
    dist_ckpt_path = init_ckpt_path / f"tp{parallel_states.get_tensor_model_parallel_rank()}"
    load_dist_ckpt = True if dist_ckpt_path.exists() else False
    ckpt_path = dist_ckpt_path if dist_ckpt_path.exists() else init_ckpt_path
    ckpt_path /= _CKPT_FILENAME

    if not ckpt_path.exists():
        logger.warning(f"Checkpoint {ckpt_path} does not exist.")
        return -1

    # load checkpoint
    timer.start("ckpt-load")
    logger.info(f"Loading checkpoint from {init_ckpt_path}")

    # Register safe globals needed to unpickle the dataclass configs stored in
    # the checkpoint under weights_only=True (the secure loading mode).
    torch.serialization.add_safe_globals(
        [
            MainConfig,
            ModelConfig,
            InitConfig,
            OptimConfig,
            DataConfig,
            ParallelConfig,
            TrainerConfig,
            OperationConfig,
            UtilsConfig,
            PositionalEmbeddingConfig,
            PEFTConfig,
            LoRAConfig,
            pathlib.PosixPath,
            pathlib.WindowsPath,
            pathlib.PurePosixPath,
            pathlib.PureWindowsPath,
        ]
    )
    checkpoint = torch.load(
        ckpt_path, weights_only=True, map_location=next(model.parameters()).device
    )

    # Unwrap DDP and torch.compile to get clean parameter names that match
    # checkpoint keys (which are always saved without wrapper prefixes).
    load_model = model
    while hasattr(load_model, "_orig_mod") or hasattr(load_model, "module"):
        if hasattr(load_model, "_orig_mod"):
            load_model = load_model._orig_mod
        if hasattr(load_model, "module"):
            load_model = load_model.module

    # load state dict
    model_attribs = {
        name: {
            "column_parallel": layer.column_parallel,
            "row_parallel": layer.row_parallel,
            "concatenated_weights": layer.concatenated_weights,
        }
        for name, layer in load_model.named_modules()
        if hasattr(layer, "column_parallel") or hasattr(layer, "row_parallel")
    }

    raw_ckpt_state = checkpoint["model_state_dict"]

    loaded_checkpoint = {}
    for name, param in load_model.named_parameters():
        loaded_param = raw_ckpt_state[name]
        module_name = ".".join(name.split(".")[:-1])

        if not load_dist_ckpt and parallel_states.get_tensor_model_parallel_world_size() > 1:
            if module_name in model_attribs:
                attribs = model_attribs[module_name]
                should_split = False

                if attribs["column_parallel"]:
                    if any(k in name for k in ["weight", "bias", "lora_B"]):
                        should_split = True
                elif attribs["row_parallel"]:
                    if any(k in name for k in ["weight", "lora_A"]):
                        should_split = True

                if should_split:
                    loaded_param = comm.split_to_model_parallel_workers(loaded_param, attribs)

        assert loaded_param is not None, f"loaded layer [{name}] is None"
        assert loaded_param.numel() == param.numel(), (
            f"loaded layer [{name}] has elements {loaded_param.numel()} which is invalid to target shape {param.shape}"
        )

        loaded_checkpoint[name] = loaded_param.reshape_as(param)

    for name, param in load_model.state_dict().items():
        if name in dict(load_model.named_parameters()):
            continue
        loaded_checkpoint[name] = raw_ckpt_state[name].reshape_as(param)

    load_model.load_state_dict(loaded_checkpoint)

    # Extract step early so we can return it even if optimizer state is missing
    last_step = checkpoint["step"]

    if config.optim.load_checkpoint_optim_state and optimizer is not None:
        loaded_optim_state_dict = checkpoint.get("optimizer_state_dict", None)

        if loaded_optim_state_dict is None:
            logger.warning(
                "Checkpoint does not contain optimizer state dict. "
                "Model weights loaded; optimizer will start fresh."
            )
        else:
            logger.info("Loading optimizer state dict.")

            loaded_optim_state = {}
            loaded_optim_state["state"] = {}
            loaded_optim_state["param_groups"] = loaded_optim_state_dict["param_groups"]

            from ironcore.offload.optimizer_helpers import _should_offload_param

            for name, param in load_model.named_parameters():
                # Skip frozen parameters (they don't have optimizer state)
                if not param.requires_grad:
                    continue

                # Skip if optimizer state doesn't exist for this param (e.g., newly added PEFT params)
                if name not in loaded_optim_state_dict["state"]:
                    logger.warning(f"Optimizer state for {name} not found in checkpoint, skipping")
                    continue

                processed_state = {}
                for state_key, state_tensor in loaded_optim_state_dict["state"][name].items():
                    if state_key in ["exp_avg", "exp_avg_sq", "max_exp_avg_sq"]:
                        # Determine target device using the same per-param criteria
                        # as the optimizer step (TP-aware via _should_offload_param).
                        offload_enabled = getattr(optimizer, "offload_enabled", False)
                        offload_min_elements = getattr(optimizer, "offload_min_param_elements", 0)
                        is_offloaded = offload_enabled and _should_offload_param(
                            param, offload_min_elements
                        )
                        target_device = torch.device("cpu") if is_offloaded else param.device

                        if state_tensor.device != target_device:
                            state_tensor = state_tensor.to(target_device)

                        # ensure param shape
                        if state_tensor.shape != param.shape:
                            try:
                                if state_tensor.shape != param.shape:
                                    state_tensor = state_tensor.reshape(param.shape).contiguous()
                            except RuntimeError as reshape_err:
                                logger.warning(
                                    f"Failed to reshape {name} from {state_tensor.shape} to {param.shape}: {reshape_err}"
                                )
                                state_tensor = None

                        if state_tensor is not None:
                            processed_state[state_key] = state_tensor
                    else:
                        processed_state[state_key] = state_tensor

                loaded_optim_state["state"][param] = processed_state

            # split optimizer state for tensor parallel
            if not load_dist_ckpt and parallel_states.get_tensor_model_parallel_world_size() > 1:
                offload_enabled = getattr(optimizer, "offload_enabled", False)
                offload_min_elements = getattr(optimizer, "offload_min_param_elements", 0)

                for name, param in load_model.named_parameters():
                    if param not in loaded_optim_state["state"]:
                        continue

                    module_name = ".".join(name.split(".")[:-1])
                    # universal checkpoint
                    optimizer_state = loaded_optim_state["state"][param]
                    for state_key in ["exp_avg", "exp_avg_sq", "max_exp_avg_sq"]:
                        if state_key not in optimizer_state:
                            continue

                        should_split = False
                        if module_name in model_attribs:
                            attribs = model_attribs[module_name]
                            if attribs["column_parallel"]:
                                if any(k in name for k in ["weight", "bias", "lora_B"]):
                                    should_split = True
                            elif attribs["row_parallel"]:
                                if any(k in name for k in ["weight", "lora_A"]):
                                    should_split = True

                        if should_split:
                            loaded_optim_state["state"][param][state_key] = (
                                comm.split_to_model_parallel_workers(
                                    optimizer_state[state_key],
                                    model_attribs[module_name],
                                )
                            )

                        tensor = loaded_optim_state["state"][param][state_key].reshape(param.shape)
                        # Keep optimizer state on CPU using same per-param criteria as step()
                        is_offloaded = offload_enabled and _should_offload_param(
                            param, offload_min_elements
                        )
                        if is_offloaded and state_key in (
                            "exp_avg",
                            "exp_avg_sq",
                            "max_exp_avg_sq",
                        ):
                            tensor = tensor.to("cpu")
                        loaded_optim_state["state"][param][state_key] = tensor

            # Handle DistributedOptimizer: partition state for local rank
            is_dist_opt = _is_distributed_optimizer(optimizer)
            if is_dist_opt and not load_dist_ckpt:
                # Universal checkpoint: partition full state for this DP rank
                loaded_optim_state = _partition_optimizer_states_for_load(
                    optimizer, loaded_optim_state, load_model
                )
                optimizer.optimizer.load_state_dict(loaded_optim_state)
            elif is_dist_opt and load_dist_ckpt:
                # Distributed checkpoint: load local partition directly
                optimizer.optimizer.load_state_dict(loaded_optim_state)
            else:
                optimizer.load_state_dict(loaded_optim_state)

    if config.optim.load_checkpoint_lr_scheduler and lr_scheduler is not None:
        lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])

    # Restore RNG state so a resumed run continues the original trajectory
    # instead of diverging. Old checkpoints (pre-fix) don't have this key.
    # (Fable issue #61.)
    rng_state = checkpoint.get("rng_state")
    if rng_state is not None:
        try:
            torch.set_rng_state(rng_state["torch_cpu"])
            if torch.cuda.is_available():
                cuda_states = rng_state.get("torch_cuda")
                if cuda_states is not None:
                    torch.cuda.set_rng_state_all(cuda_states)
            py_rs = rng_state.get("python_random")
            if py_rs is not None:
                random.setstate(py_rs)
            np_rs = rng_state.get("numpy")
            if np_rs is not None:
                try:
                    import numpy as np

                    keys = (
                        np_rs["keys"].numpy() if hasattr(np_rs["keys"], "numpy") else np_rs["keys"]
                    )
                    np.random.set_state(
                        (
                            np_rs["name"],
                            keys,
                            np_rs["pos"],
                            np_rs["has_gauss"],
                            np_rs["cached_gaussian"],
                        )
                    )
                except (ImportError, KeyError, TypeError) as rng_err:
                    logger.warning(f"Could not restore numpy RNG state: {rng_err}")
        except (KeyError, TypeError, RuntimeError) as rng_err:
            logger.warning(f"Could not restore RNG state from checkpoint: {rng_err}")

    timer.stop("ckpt-load")
    logger.info(
        f"Checkpoint loaded successfully. Resuming training at step {last_step}. Total time: {timer.get('ckpt-load'):.2f}s"
    )

    return last_step


def save_checkpoint(
    config: MainConfig,
    model: LanguageModel,
    optimizer: torch.optim.Optimizer,
    lr_scheduler: LRScheduler,
    step: int,
):
    """Save a checkpoint."""

    logger = get_logger()
    timer = get_timer()

    if config.operation.no_save:
        if config.trainer.model_path == "":
            logger.info("Skip checkpoint saving due to the unspecified model path")
        else:
            logger.info("Skip checkpoint saving since no-save flag is set")
        return

    assert config.trainer.model_path, (
        "trainer.model_path is not set. "
        "Specify a checkpoint save directory in config, or set operation.no_save: true."
    )

    # checkpoint file name
    init_ckpt_path = Path(config.trainer.model_path) / f"step_{step}"
    ckpt_path = (
        init_ckpt_path / f"tp{parallel_states.get_tensor_model_parallel_rank()}"
        if config.operation.save_dist_ckpt
        else init_ckpt_path
    )
    ckpt_path /= _CKPT_FILENAME

    timer.start("ckpt-save")

    if not ckpt_path.parent.exists():
        ckpt_path.parent.mkdir(parents=True, exist_ok=True)

    def _is_universal_checkpoint(config: MainConfig):
        """checking requested checkpoint format"""
        return (
            not config.operation.save_dist_ckpt
            and parallel_states.get_tensor_model_parallel_world_size() > 1
        )

    # Unwrap DDP and torch.compile for clean parameter names
    save_model = model
    while hasattr(save_model, "_orig_mod") or hasattr(save_model, "module"):
        if hasattr(save_model, "_orig_mod"):
            save_model = save_model._orig_mod
        if hasattr(save_model, "module"):
            save_model = save_model.module

    # model_state_dict
    model_attribs = {
        name: {
            "column_parallel": layer.column_parallel,
            "row_parallel": layer.row_parallel,
            "concatenated_weights": layer.concatenated_weights,
        }
        for name, layer in save_model.named_modules()
        if hasattr(layer, "column_parallel") or hasattr(layer, "row_parallel")
    }

    model_state_dict = {}
    for name, param in save_model.state_dict().items():
        module_name = ".".join(name.split(".")[:-1])

        output_param = param
        if _is_universal_checkpoint(config):
            if module_name in model_attribs:
                attribs = model_attribs[module_name]
                should_gather = False

                if attribs["column_parallel"]:
                    if any(k in name for k in ["weight", "bias", "lora_B"]):
                        should_gather = True
                elif attribs["row_parallel"]:
                    if any(k in name for k in ["weight", "lora_A"]):
                        should_gather = True

                if should_gather:
                    output_param = comm.gather_from_model_parallel_workers(param, attribs)
                else:
                    # Replicated - only save from rank 0
                    if parallel_states.get_tensor_model_parallel_rank() == 0:
                        output_param = param
                    else:
                        output_param = param  # Will only be saved from rank 0 anyway
            else:
                # Replicated or unknown - only save from rank 0
                if parallel_states.get_tensor_model_parallel_rank() == 0:
                    output_param = param
                else:
                    output_param = param

        model_state_dict[name] = output_param

    # When PEFT (LoRA) is active, save only adapter weights by default. Base
    # weights are frozen and can be reloaded from the pretrained checkpoint,
    # so including them bloats the file ~100x and prevents the artifact from
    # being distributed as a standalone adapter. An explicit
    # operation.save_full_model opt-in restores the previous behaviour.
    # (Fable issue #65.)
    is_peft_active = any("lora_" in name for name, _ in save_model.named_parameters())
    if is_peft_active and not getattr(config.operation, "save_full_model", False):
        adapter_only = {name: t for name, t in model_state_dict.items() if "lora_" in name}
        if adapter_only:
            model_state_dict = adapter_only
            logger.info(
                f"PEFT active: saving adapter-only checkpoint "
                f"({len(model_state_dict)} tensors). Set operation.save_full_model=true "
                f"to include base weights."
            )

    # optimizer state - build dict keyed by parameter name (not integer index)
    is_dist_opt = _is_distributed_optimizer(optimizer)

    if is_dist_opt and not config.operation.save_dist_ckpt:
        # DistributedOptimizer with universal checkpoint: gather from all DP ranks
        dp_group = parallel_states.get_data_parallel_group()
        optimizer_state_dict_by_name = _gather_distributed_optimizer_states(
            optimizer, save_model, dp_group
        )
    else:
        # Standard optimizer or distributed checkpoint: use local state
        optimizer_state_dict_by_name = {
            "state": {},
            "param_groups": optimizer.state_dict()["param_groups"],
        }
        # Frozen parameters (e.g. LoRA base_layer weights) were never registered
        # with the optimizer. optimizer.state is a defaultdict, so indexing it
        # with a param that was never registered silently inserts a stray
        # entry into the live optimizer's state — corrupting it for any later
        # optimizer.state_dict() call (e.g. the universal-checkpoint merge
        # below, or the next checkpoint save).
        for _i, (name, param) in enumerate(save_model.named_parameters()):
            if not param.requires_grad:
                continue
            optimizer_state_dict_by_name["state"][name] = optimizer.state[param]

    # For universal checkpoints, gather TP-sharded optimizer states
    final_optimizer_state = optimizer_state_dict_by_name
    if _is_universal_checkpoint(config):
        merged_optimizer_state = {
            "state": {},
            "param_groups": optimizer.state_dict()["param_groups"],
        }

        # Frozen parameters (e.g. LoRA base_layer weights) are never registered
        # with the optimizer, so optimizer.state_dict()["state"] only has one
        # entry per *trainable* parameter. Filter before zipping with
        # strict=True, rather than after — zip(..., strict=True) raises on a
        # length mismatch before the loop body ever runs.
        trainable_named_parameters = [
            (name, param) for name, param in save_model.named_parameters() if param.requires_grad
        ]
        for _i, ((name, param), optim_state_id) in enumerate(
            zip(trainable_named_parameters, optimizer.state_dict()["state"], strict=True)
        ):
            module_name = ".".join(name.split(".")[:-1])
            optim_state = optimizer.state_dict()["state"][optim_state_id]

            output_optim_state = {}
            for key in ["exp_avg", "exp_avg_sq", "max_exp_avg_sq"]:
                if key not in optim_state:
                    continue
                should_gather = False
                if module_name in model_attribs:
                    attribs = model_attribs[module_name]
                    if attribs["column_parallel"]:
                        if any(k in name for k in ["weight", "bias", "lora_B"]):
                            should_gather = True
                    elif attribs["row_parallel"]:
                        if any(k in name for k in ["weight", "lora_A"]):
                            should_gather = True

                if should_gather:
                    output_optim_state[key] = comm.gather_from_model_parallel_workers(
                        optim_state[key], model_attribs[module_name]
                    )
                else:
                    output_optim_state[key] = optim_state[key]
            output_optim_state["step"] = step

            # Use parameter name as key (not integer index) for consistent load format
            merged_optimizer_state["state"][name] = output_optim_state

        final_optimizer_state = merged_optimizer_state

    # HuggingFace compatible config (optional — only if hf_model_type/hf_architecture set)
    hf_config = None
    if config.model.hf_model_type is not None and config.model.hf_architecture is not None:
        hf_config = HFConfigManager.get_hf_config(config)
    else:
        logger.info("Skipping HF config: hf_model_type/hf_architecture not set")

    # Convert dataclass configs to dicts for safe serialization (weights_only=True compatible)
    model_config_dict = None
    if hasattr(model, "config"):
        if dataclasses.is_dataclass(model.config):
            model_config_dict = dataclasses.asdict(model.config)
        else:
            model_config_dict = model.config

    hf_config_dict = None
    if hf_config is not None:
        if dataclasses.is_dataclass(hf_config):
            hf_config_dict = dataclasses.asdict(hf_config)
        else:
            hf_config_dict = hf_config

    logger.info(f"Saving checkpoint to {str(init_ckpt_path)}")
    # Capture full RNG state so a resumed run continues the original trajectory
    # instead of diverging. (Fable issue #61.) np.random.get_state() returns
    # a tuple whose second element is an ndarray — convert to a (dtype, bytes)
    # pair so weights_only=True loading works without registering numpy
    # internals as safe globals.
    np_state = None
    try:
        import numpy as np

        raw = np.random.get_state()
        np_state = {
            "name": raw[0],
            "keys": torch.as_tensor(raw[1]),
            "pos": int(raw[2]),
            "has_gauss": int(raw[3]),
            "cached_gaussian": float(raw[4]),
        }
    except ImportError:
        pass

    checkpoint = {
        "model_state_dict": model_state_dict,
        "optimizer_state_dict": final_optimizer_state,
        "lr_scheduler": lr_scheduler.state_dict(),
        "step": step,
        "config": model_config_dict,
        "hf_config": hf_config_dict,
        "rng_state": {
            "torch_cpu": torch.get_rng_state(),
            "torch_cuda": torch.cuda.get_rng_state_all(),
            "numpy": np_state,
            "python_random": random.getstate(),
        },
    }

    # save checkpoint — atomic write via .tmp + fsync + os.replace so an
    # interrupted save cannot leave a truncated pytorch_model.bin that would
    # permanently break resume. (Fable issue #58.)
    if parallel_states.get_data_parallel_group_rank() == 0 and (
        config.operation.save_dist_ckpt or parallel_states.get_tensor_model_parallel_rank() == 0
    ):
        tmp_path = ckpt_path.with_suffix(ckpt_path.suffix + ".tmp")
        with open(tmp_path, "wb") as f:
            torch.save(checkpoint, f)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, ckpt_path)

    timer.stop("ckpt-save")
    # Barrier BEFORE writing latest_step.txt so readers never see a step
    # number pointing at a half-written TP shard on another rank.
    if parallel_states.get_tensor_model_parallel_world_size() > 1:
        dist.barrier(group=parallel_states.get_tensor_model_parallel_group())

    # latest_step.txt — atomic, written after all shard writers finished.
    if is_first_rank():
        with open(
            Path(config.trainer.model_path) / _LATEST_STEP_FILENAME,
            "w",
            encoding="utf-8",
        ) as f:
            f.write(f"{step}\n")
            f.flush()
            os.fsync(f.fileno())

        # Save HuggingFace compatible config (only if HF fields are set)
        if config.model.hf_model_type is not None and config.model.hf_architecture is not None:
            HFConfigManager.save_hf_config(config, config.trainer.model_path)

    logger.info(f"Checkpoint saved successfully. Checkpoint saved in {timer.get('ckpt-save'):.3f}s")
