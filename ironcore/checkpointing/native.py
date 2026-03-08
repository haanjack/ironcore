# Copyright (c) 2025-2026 Jaegeun Han
#
# SPDX-License-Identifier: Apache-2.0

import dataclasses
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

    # Register safe globals for future use with weights_only=True
    # Note: Currently using weights_only=False to support optimizer states
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
        ]
    )
    checkpoint = torch.load(
        ckpt_path, weights_only=True, map_location=next(model.parameters()).device
    )

    # Load config
    checkpoint_config = checkpoint.get("config")
    if checkpoint_config is not None:
        HFConfigManager.load_hf_config(Path(config.trainer.model_path))
        logger.info("Model configuration loaded from checkpoint.")
    else:
        logger.warning("No model configuration found in checkpoint.")

    # load state dict
    model_attribs = {
        name: {
            "column_parallel": layer.column_parallel,
            "row_parallel": layer.row_parallel,
            "concatenated_weights": layer.concatenated_weights,
        }
        for name, layer in model.named_modules()
        if hasattr(layer, "column_parallel") or hasattr(layer, "row_parallel")
    }

    # Normalize checkpoint state dict to match model's parameter namespace.
    # Handles the case where the checkpoint was saved from a DDP-wrapped model
    # (keys have "module." prefix) but is being loaded into a non-DDP model
    # (or vice versa).
    raw_ckpt_state = checkpoint["model_state_dict"]
    model_param_names = {n for n, _ in model.named_parameters()}
    ckpt_keys = set(raw_ckpt_state.keys())
    if not ckpt_keys.issubset(model_param_names | set(model.state_dict())):
        model_has_module = any(n.startswith("module.") for n in model_param_names)
        ckpt_has_module = any(k.startswith("module.") for k in ckpt_keys)
        if ckpt_has_module and not model_has_module:
            raw_ckpt_state = {k[len("module.") :]: v for k, v in raw_ckpt_state.items()}
        elif not ckpt_has_module and model_has_module:
            raw_ckpt_state = {f"module.{k}": v for k, v in raw_ckpt_state.items()}

    loaded_checkpoint = {}
    for name, param in model.named_parameters():
        loaded_param = raw_ckpt_state[name]
        module_name = ".".join(name.split(".")[:-1])

        if not load_dist_ckpt and parallel_states.get_tensor_model_parallel_world_size() > 1:
            if module_name in model_attribs:
                attribs = model_attribs[module_name]
                should_split = False

                if attribs["column_parallel"]:
                    # Split if it's base weight/bias or LoRA B
                    if any(k in name for k in ["weight", "bias", "lora_B"]):
                        should_split = True
                elif attribs["row_parallel"]:
                    # Split if it's base weight or LoRA A
                    if any(k in name for k in ["weight", "lora_A"]):
                        should_split = True

                if should_split:
                    loaded_param = comm.split_to_model_parallel_workers(loaded_param, attribs)

        # Sanity check
        assert loaded_param is not None, f"loaded layer [{name}] is None"

        # assert torch.all(param_ == get_tensor_model_parallel_rank()), f"loaded state {name} are not aligned with tensor model parallel"
        assert loaded_param.numel() == param.numel(), (
            f"loaded layer [{name}] has elements {loaded_param.numel()} which is invalid to target shape {param.shape}"
        )

        loaded_checkpoint[name] = loaded_param.reshape_as(param)

    for name, param in model.state_dict().items():
        if name in dict(model.named_parameters()):
            continue
        loaded_checkpoint[name] = raw_ckpt_state[name].reshape_as(param)

    model.load_state_dict(loaded_checkpoint)

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

            for name, param in model.named_parameters():
                # Skip frozen parameters (they don't have optimizer state)
                if not param.requires_grad:
                    continue

                # Skip if optimizer state doesn't exist for this param (e.g., newly added PEFT params)
                if name not in loaded_optim_state_dict["state"]:
                    logger.warning(f"Optimizer state for {name} not found in checkpoint, skipping")
                    continue

                processed_state = {}
                for state_key, state_tensor in loaded_optim_state_dict["state"][name].items():
                    if state_key in ["exp_avg", "exp_avg_sq"]:
                        # ensure device matches
                        if state_tensor.device != param.device:
                            state_tensor = state_tensor.to(param.device)

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
                for name, param in model.named_parameters():
                    if param not in loaded_optim_state["state"]:
                        continue

                    module_name = ".".join(name.split(".")[:-1])
                    # universal checkpoint
                    optimizer_state = loaded_optim_state["state"][param]
                    for state_key in ["exp_avg", "exp_avg_sq"]:
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

                        loaded_optim_state["state"][param][state_key] = loaded_optim_state["state"][
                            param
                        ][state_key].reshape(param.shape)

            optimizer.load_state_dict(loaded_optim_state)

    if config.optim.load_checkpoint_lr_scheduler and lr_scheduler is not None:
        lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])

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

    model_attribs = {
        name: {
            "column_parallel": layer.column_parallel,
            "row_parallel": layer.row_parallel,
            "concatenated_weights": layer.concatenated_weights,
        }
        for name, layer in model.named_modules()
        if hasattr(layer, "column_parallel") or hasattr(layer, "row_parallel")
    }

    # model_state_dict
    model_state_dict = {}
    for name, param in model.state_dict().items():
        # remove 'weights' or 'bias' from the name
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

    # optimizer state - build dict keyed by parameter name (not integer index)
    optimizer_state_dict_by_name = {
        "state": {},
        "param_groups": optimizer.state_dict()["param_groups"],
    }
    for _i, (name, param) in enumerate(model.named_parameters()):
        optimizer_state_dict_by_name["state"][name] = optimizer.state[param]

    # For universal checkpoints, gather TP-sharded optimizer states
    final_optimizer_state = optimizer_state_dict_by_name
    if _is_universal_checkpoint(config):
        merged_optimizer_state = {
            "state": {},
            "param_groups": optimizer.state_dict()["param_groups"],
        }

        for _i, ((name, param), optim_state_id) in enumerate(
            zip(model.named_parameters(), optimizer.state_dict()["state"], strict=True)
        ):
            # Skip frozen parameters
            if not param.requires_grad:
                continue

            module_name = ".".join(name.split(".")[:-1])
            optim_state = optimizer.state_dict()["state"][optim_state_id]

            output_optim_state = {}
            for key in ["exp_avg", "exp_avg_sq"]:
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

    # HuggingFace compatible config
    hf_config = HFConfigManager.get_hf_config(config)

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
    checkpoint = {
        "model_state_dict": model_state_dict,
        "optimizer_state_dict": final_optimizer_state,
        "lr_scheduler": lr_scheduler.state_dict(),
        "step": step,
        "config": model_config_dict,
        "hf_config": hf_config_dict,
    }

    # save checkpoint
    if parallel_states.get_data_parallel_group_rank() == 0 and (
        config.operation.save_dist_ckpt or parallel_states.get_tensor_model_parallel_rank() == 0
    ):
        with open(ckpt_path, "wb") as f:
            torch.save(checkpoint, f)

    # save latest_step file
    if is_first_rank():
        with open(
            Path(config.trainer.model_path) / _LATEST_STEP_FILENAME,
            "w",
            encoding="utf-8",
        ) as f:
            f.write(f"{step}\n")

        # Save HuggingFace compatible config
        HFConfigManager.save_hf_config(config, config.trainer.model_path)

    timer.stop("ckpt-save")
    if parallel_states.get_tensor_model_parallel_world_size() > 1:
        dist.barrier(group=parallel_states.get_tensor_model_parallel_group())
    logger.info(f"Checkpoint saved successfully. Checkpoint saved in {timer.get('ckpt-save'):.3f}s")
