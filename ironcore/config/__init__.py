# Copyright (c) 2025-2026 Jaegeun Han
#
# SPDX-License-Identifier: Apache-2.0

import os
from argparse import Namespace
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

from ironcore.utils import load_yaml_config
from ironcore.utils.config import sanitize_path_component as _sanitize_path_component
from ironcore.utils.config import validate_path_within_dir as _validate_path_within_dir

from .config import BaseConfig
from .config_alignment import AlignmentConfig
from .config_data import DataConfig
from .config_model import KVCacheConfig as KVCacheConfig
from .config_model import ModelConfig
from .config_model import PositionalEmbeddingConfig as PositionalEmbeddingConfig
from .config_offload import OffloadConfig
from .config_optim import OptimConfig
from .config_parallel import ParallelConfig
from .config_peft import LoRAConfig as LoRAConfig
from .config_peft import PEFTConfig
from .config_trainer import InitConfig, OperationConfig, TrainerConfig
from .config_utils import ProfilerConfig, UtilsConfig


@dataclass
class MainConfig(BaseConfig):
    """trainer configuration."""

    model: ModelConfig
    init: InitConfig
    optim: OptimConfig
    data: DataConfig
    parallel: ParallelConfig
    trainer: TrainerConfig
    operation: OperationConfig
    utils: UtilsConfig
    profiler: ProfilerConfig
    peft: PEFTConfig
    alignment: AlignmentConfig = field(default_factory=AlignmentConfig)
    offload: OffloadConfig = field(default_factory=OffloadConfig)


def _config_validation(config: MainConfig):
    """Validate arguments and update internal enum if necessary"""
    # train steps — 0 selects evaluation-only mode (used by `ironcore evaluate`)
    if config.operation.train_steps < 0:
        raise ValueError("operation.train_steps should not be negative")

    dp_group_size = config.trainer.tensor_model_parallel_size
    dp_world_size = config.parallel.world_size // dp_group_size
    if dp_world_size <= 0:
        raise ValueError(
            f"World size ({config.parallel.world_size}) is smaller than single data parallelism group size ({dp_group_size})"
        )

    # batch size validation
    if [
        config.trainer.micro_batch_size,
        config.trainer.train_batch_size,
        config.trainer.gradient_accumulation_steps,
    ].count(None) == 1:
        if config.trainer.train_batch_size is None:
            config.trainer.train_batch_size = (
                config.trainer.micro_batch_size
                * config.trainer.gradient_accumulation_steps
                * dp_world_size
            )
        elif config.trainer.gradient_accumulation_steps is None:
            config.trainer.gradient_accumulation_steps = (
                config.trainer.train_batch_size // config.trainer.micro_batch_size // dp_world_size
            )
        elif config.trainer.micro_batch_size is None:
            config.trainer.micro_batch_size = (
                config.trainer.train_batch_size
                // config.trainer.gradient_accumulation_steps
                // dp_world_size
            )

        if config.trainer.train_batch_size % (config.trainer.micro_batch_size * dp_world_size) != 0:
            raise ValueError(
                "train_batch_size should be divisible by micro_batch_size * data parallelism group size"
            )
    elif [
        config.trainer.micro_batch_size,
        config.trainer.train_batch_size,
        config.trainer.gradient_accumulation_steps,
    ].count(None) > 1:
        raise ValueError(
            "micro_batch_size, train_batch_size, gradient_accumulation_steps are not sufficiently specified"
        )

    if (
        config.trainer.micro_batch_size * config.trainer.gradient_accumulation_steps * dp_world_size
        != config.trainer.train_batch_size
    ):
        raise ValueError(
            "micro_batch_size * gradient_accumulation_steps should be equal to train_batch_size"
        )

    # model parallel validation
    if config.trainer.tensor_model_parallel_size > 1 and config.model.name != "dummy":
        if config.model.num_attention_heads % config.trainer.tensor_model_parallel_size != 0:
            raise ValueError(
                "num_attention_heads should be divisible by tensor_model_parallel_size"
            )
        if config.model.num_attention_groups % config.trainer.tensor_model_parallel_size != 0:
            raise ValueError(
                "num_attention_groups should be divisible by tensor_model_parallel_size"
            )

    # positional embedding
    if config.model.positional_embedding.type.lower() not in [
        "absolute",
        "rope",
        "none",
    ]:
        raise ValueError("Available positional embedding options are ['absolute', 'rope', 'none'].")

    import torch

    if torch.cuda.device_count() == 0 and config.trainer.tensor_model_parallel_size > 1:
        raise ValueError("tensor_model_parallel_size should be 1 in non-CUDA environments")

    # DistributedOptimizer validation
    if config.parallel.use_distributed_optimizer:
        if config.parallel.use_fsdp:
            raise ValueError(
                "use_distributed_optimizer is incompatible with FSDP. "
                "Use FSDP's built-in sharding (fsdp_sharding_strategy) instead."
            )
        if dp_world_size <= 1:
            import warnings

            warnings.warn(
                "use_distributed_optimizer is enabled but DP world size is 1. "
                "No optimizer state partitioning will occur. "
                "Increase world_size or decrease tensor_model_parallel_size to enable partitioning.",
                stacklevel=2,
            )

    # Offload validation
    if config.offload.optimizer_offload and not config.offload.enabled:
        raise ValueError("offload.optimizer_offload requires offload.enabled to be true")
    if config.offload.optimizer_state_precision not in ("fp32", "fp16", "bf16"):
        raise ValueError(
            f"offload.optimizer_state_precision must be fp32, fp16, or bf16, "
            f"got '{config.offload.optimizer_state_precision}'"
        )
    if config.offload.activation_spill and not config.offload.enabled:
        raise ValueError("offload.activation_spill requires offload.enabled to be true")
    if config.offload.activation_spill_granularity not in ("sub_layer", "full_layer"):
        raise ValueError(
            f"offload.activation_spill_granularity must be 'sub_layer' or 'full_layer', "
            f"got '{config.offload.activation_spill_granularity}'"
        )
    if config.offload.activation_spill and config.operation.activation_recompute:
        import warnings

        warnings.warn(
            "offload.activation_spill is enabled but activation_recompute is also enabled. "
            "Activation spilling replaces checkpointing. Disabling activation_recompute.",
            stacklevel=2,
        )
        config.operation.activation_recompute = False
    if config.offload.weight_offload and not config.offload.enabled:
        raise ValueError("offload.weight_offload requires offload.enabled to be true")
    if config.offload.weight_offload and config.offload.weight_storage_precision not in (
        "fp32",
        "fp16",
        "bf16",
    ):
        raise ValueError(
            f"offload.weight_storage_precision must be fp32, fp16, or bf16, "
            f"got '{config.offload.weight_storage_precision}'"
        )
    if config.offload.weight_offload and config.parallel.use_fsdp:
        raise ValueError(
            "offload.weight_offload is incompatible with FSDP. "
            "FSDP manages its own parameter sharding/unsharding."
        )
    if config.offload.weight_offload and not config.offload.activation_spill:
        import warnings

        warnings.warn(
            "offload.weight_offload requires activation spilling for weight "
            "eviction (no_autograd_graph). Enabling offload.activation_spill automatically.",
            stacklevel=2,
        )
        config.offload.activation_spill = True
    if config.offload.weight_prefetch_layers < 1:
        raise ValueError(
            f"offload.weight_prefetch_layers must be >= 1, got {config.offload.weight_prefetch_layers}"
        )

    # Optimizer offload + FSDP FULL_SHARD → ValueError (host OOM from duplicated optimizer states)
    if config.offload.optimizer_offload and config.parallel.use_fsdp:
        if config.parallel.fsdp_sharding_strategy == "full":
            raise ValueError(
                "offload.optimizer_offload + FSDP full_shard duplicates optimizer states on host. "
                "Use fsdp_sharding_strategy: shard_grad_op or disable optimizer_offload."
            )

    # Optimizer offload + FSDP CPUOffload → ValueError (redundant)
    if config.offload.optimizer_offload and config.parallel.use_fsdp:
        if config.parallel.fsdp_offload_params:
            raise ValueError("optimizer_offload is redundant with FSDP CPUOffload. Use only one.")

    # Optimizer offload + FSDP without use_orig_params → ValueError (FlatParameter breaks optimizer refs)
    if config.offload.optimizer_offload and config.parallel.use_fsdp:
        if not config.parallel.fsdp_use_orig_params:
            raise ValueError(
                "FSDP + optimizer_offload requires fsdp_use_orig_params=True. "
                "Without it, FSDP replaces parameters with FlatParameter, breaking optimizer references."
            )

    # TP + offload validation
    if config.offload.enabled and config.offload.weight_offload:
        if config.trainer.tensor_model_parallel_size > 1 and dp_world_size > 1:
            raise ValueError(
                f"Weight streaming with tensor_parallel={config.trainer.tensor_model_parallel_size} "
                f"and data_parallel={dp_world_size} is not supported. "
                "ZeRO-3 sharding only works with TP=1 + DP>1 or TP>1 + DP=1. "
                "Use TP-only (DP=1) or TP=1 + DP>1."
            )
        if config.parallel.use_distributed_optimizer:
            raise ValueError(
                "weight_offload is incompatible with use_distributed_optimizer. "
                "DistributedOptimizer broadcasts parameters via NCCL, which requires GPU tensors. "
                "With weight_offload, parameters live on CPU. "
                "Disable use_distributed_optimizer or disable weight_offload."
            )
        if dp_world_size > 1 and config.parallel.use_fsdp:
            raise ValueError(
                "weight_offload + DP>1 is incompatible with FSDP. "
                "Remove use_fsdp to use ZeRO-3 offload sharding."
            )
        if config.trainer.tensor_model_parallel_size > 1:
            import logging

            logging.getLogger("ironcore.config").info(
                "TP + offload enabled: each rank will stream its own TP shard. "
                "Embedding and output head stay on GPU for TP communication."
            )
        if dp_world_size > 1:
            import logging

            logging.getLogger("ironcore.config").info(
                f"ZeRO-3 parameter sharding enabled: weight_offload + DP={dp_world_size}. "
                "Parameters sharded across DP ranks, all-gather on GPU via NCCL."
            )

    # Auto-detect pinned memory pool size if requested (-1.0)
    if config.offload.enabled and config.offload.pinned_memory_pool_gb == -1.0:
        import logging
        import warnings

        from ironcore.utils import available_host_memory_gb, total_host_memory_gb

        # Simple auto-detection: 40% of available RAM, min 8GB
        avail = available_host_memory_gb()
        auto_size = avail * 0.40
        auto_size = max(8.0, auto_size)
        config.offload.pinned_memory_pool_gb = auto_size

        logging.info(
            f"Auto-detected pinned memory pool: {auto_size:.1f}GB "
            f"(from {avail:.1f}GB available / {total_host_memory_gb():.1f}GB total)"
        )

    # Resolve optimizer CPU thread count
    if config.offload.enabled and (
        config.offload.optimizer_offload or config.offload.weight_offload
    ):
        if config.offload.optimizer_cpu_threads == -1:
            import os

            total_cores = os.cpu_count() or 1
            auto_threads = max(1, int(total_cores * 0.8))
            config.offload.optimizer_cpu_threads = auto_threads

    # Warn if pinned pool size exceeds 80% of total RAM
    if config.offload.enabled and config.offload.pinned_memory_pool_gb > 0:
        import warnings

        from ironcore.utils import total_host_memory_gb

        total_ram = total_host_memory_gb()
        if config.offload.pinned_memory_pool_gb > 0.8 * total_ram:
            warnings.warn(
                f"offload.pinned_memory_pool_gb ({config.offload.pinned_memory_pool_gb:.1f}GB) "
                f"exceeds 80% of total system RAM ({total_ram:.1f}GB). "
                f"Risk of host OOM. Consider using pinned_memory_pool_gb: -1.0 for auto-sizing.",
                stacklevel=2,
            )


def load_data_config(config, datasets: dict[str, Any]) -> list[dict[str, Any]]:
    """build data config."""

    output_list = []
    for dataset_name_or_path, dataset_config in datasets.items():
        # setup dataset config
        # - [train, eval, test]
        #   - dataset_name_or_path
        #     - content_column: text
        #     - subgroup: dataset subgroup name
        #     - ratio: 1.0

        loaded_config = {}

        # check if dataset_name_or_path is a path or a name
        if Path(dataset_name_or_path).with_suffix(".bin").exists():
            # if dataset_name_or_path is a path
            loaded_config["name"] = os.path.basename(dataset_name_or_path)
            loaded_config["dataset_path"] = dataset_name_or_path
        else:
            # if dataset_name_or_path is a name
            if os.environ.get("PROCESSED_DATA_PATH"):
                base_dir = os.environ.get("PROCESSED_DATA_PATH")
            elif hasattr(config.data, "preprocessed_dir") and config.data.preprocessed_dir:
                base_dir = Path(config.data.preprocessed_dir)
            else:
                base_dir = Path("data/preprocessed_corpus")

            dataset_name = dataset_name_or_path
            if dataset_config.get("subgroup", None):
                dataset_name = f"{dataset_name}_{dataset_config.get('subgroup')}"
            loaded_config["name"] = dataset_name
            loaded_config["dataset_path"] = base_dir / config.data.name / dataset_name

        if "ratio" in dataset_config:
            loaded_config["ratio"] = dataset_config.get("ratio", 1.0)
        if "samples" in dataset_config:
            loaded_config["samples"] = dataset_config.get("samples")

        output_list.append(loaded_config)

    return output_list


def _update_config_from_yaml(config: dataclass, config_group_key: str, config_group: dict):
    """update config from yaml config file."""

    # get config from yaml
    config_dict_item = asdict(config)[config_group_key]
    for yaml_config_key, yaml_config_value in config_group.items():
        if yaml_config_key not in config_dict_item:
            raise ValueError(
                f"{yaml_config_key} is not defined in {config_group_key} config. Check yaml config file."
            )
        config_dict_item[yaml_config_key] = yaml_config_value

    # update config
    getattr(config, config_group_key)(**config_dict_item)


def _update_data_config_from_yaml(config: dataclass, config_group_key, config_group: dict):
    """update data config from yaml config file."""
    # New-format configs use train_datasets / eval_datasets / test_datasets keys.
    # Delegate to DataConfig's own parser which already understands this format.
    if any(k in config_group for k in ("train_datasets", "eval_datasets", "test_datasets")):
        parsed = DataConfig._parse_config_dict(config_group)
        for f in fields(parsed):
            setattr(config.data, f.name, getattr(parsed, f.name))
        return

    for sub_group_key, sub_group_value in config_group.items():
        if sub_group_key in ["train", "eval", "test"]:
            sub_group_key = f"{sub_group_key}_datasets"
            setattr(config.data, sub_group_key, load_data_config(config, sub_group_value))
        else:
            # update arguments for DataConfig class
            if sub_group_key not in config_group:
                raise ValueError(
                    f"{sub_group_key} is not defined in {config_group_key}. Check yaml config file."
                )
            setattr(config.data, sub_group_key, sub_group_value)


def _load_subgroup_config_from_yaml(config, config_group_key, sub_group_config):
    """
    Load subgroup config
    """
    if config_group_key == "data":
        _update_data_config_from_yaml(config, config_group_key, sub_group_config)
    else:
        # load configs: trainer, optimizer, model, etc
        _update_config_from_yaml(config, config_group_key, sub_group_config)


def _load_config_from_yaml(config: dataclass, args: Namespace):
    """
    Load config from yaml config file.

    yaml config file can accept several pre-defined groups. Those groups are predefined in MainConfig class.

    yaml config file can have two type of format.

    (type 1) - listing arguments in yaml config file
    [config-group]:
        [argument 1]: value
        [argument 2]: value

    (type 2) - external yaml config file
    [config-group]: config_name

    - when config group is specified with a string, trainer finds external config file as 'configs/[config-group]/config_name.yaml'.


    """

    yaml_config = load_yaml_config(args.config_path)
    config_dict = asdict(config)
    # load configs from yaml
    for config_group_key, sub_group_config in yaml_config.items():
        # check if config group is defined
        if config_group_key not in config_dict:
            raise ValueError(f"{config_group_key} is not defined configuration group")

        # load configs from yaml
        if isinstance(sub_group_config, dict) and (
            "config-path" in sub_group_config or "config_path" in sub_group_config
        ):
            # Explicit form: model: {name: foo, config-path: configs/model/foo.yaml, <overrides>}
            # Both config-path and config_path are accepted for consistency with other config keys.
            # config-path is resolved relative to the working directory.
            config_path_key = "config-path" if "config-path" in sub_group_config else "config_path"
            explicit_path = Path(sub_group_config[config_path_key])
            if not explicit_path.exists():
                raise FileNotFoundError(f"Config file not found: {explicit_path}")

            name = sub_group_config.get("name")
            if name:
                sub_config = getattr(config, config_group_key)
                if hasattr(sub_config, "name"):
                    sub_config.name = name

            sub_group_config_from_file = load_yaml_config(explicit_path)
            _load_subgroup_config_from_yaml(config, config_group_key, sub_group_config_from_file)

            # Apply any inline overrides (keys other than name / config-path / config_path)
            overrides = {
                k: v
                for k, v in sub_group_config.items()
                if k not in ("name", "config-path", "config_path")
            }
            if overrides:
                _load_subgroup_config_from_yaml(config, config_group_key, overrides)

        elif isinstance(sub_group_config, str):
            # Short-form string reference: model: nanogpt-small
            # Resolves to configs/{group}/{name}.yaml relative to the config file.
            getattr(config, config_group_key).name = sub_group_config

            sanitized_config_name = _sanitize_path_component(sub_group_config)

            config_base_dir = Path(args.config_path).parent
            sub_group_config_path = (
                config_base_dir / f"{config_group_key}/{sanitized_config_name}.yaml"
            )
            # Walk up one level for configs living in a subdirectory (e.g. configs/experiments/)
            if not sub_group_config_path.exists() and config_base_dir.parent != config_base_dir:
                fallback_base = config_base_dir.parent
                fallback_path = fallback_base / f"{config_group_key}/{sanitized_config_name}.yaml"
                if fallback_path.exists():
                    config_base_dir = fallback_base
                    sub_group_config_path = fallback_path

            if not _validate_path_within_dir(sub_group_config_path, config_base_dir):
                raise ValueError(
                    f"Invalid config path: {sub_group_config} resolves outside the config directory"
                )

            if sanitized_config_name == "dummy":
                if sub_group_config_path.exists():
                    sub_group_config_from_file = load_yaml_config(sub_group_config_path)
                continue
            else:
                if not sub_group_config_path.exists():
                    raise FileNotFoundError(f"Config file not found: {sub_group_config_path}")
                sub_group_config_from_file = load_yaml_config(sub_group_config_path)

            _load_subgroup_config_from_yaml(config, config_group_key, sub_group_config_from_file)

        else:
            _load_subgroup_config_from_yaml(config, config_group_key, sub_group_config)


def print_args(config):
    """Print config."""
    for group_field in fields(config):
        for field_ in fields(group_field.type):
            print(f"{field_.name}: {getattr(config, field_.name)}")
