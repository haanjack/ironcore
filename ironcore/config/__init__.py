# Copyright (c) 2025-2026 Jaegeun Han
#
# SPDX-License-Identifier: Apache-2.0

import os
from argparse import ArgumentParser, Namespace
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Optional, Union, get_args, get_origin

import torch
from dotenv import load_dotenv

from ironcore.utils import load_yaml_config

from .config import BaseConfig
from .config_alignment import AlignmentConfig
from .config_data import DataConfig
from .config_model import KVCacheConfig as KVCacheConfig
from .config_model import ModelConfig
from .config_model import PositionalEmbeddingConfig as PositionalEmbeddingConfig
from .config_optim import OptimConfig
from .config_parallel import ParallelConfig
from .config_peft import LoRAConfig as LoRAConfig
from .config_peft import PEFTConfig
from .config_trainer import InitConfig, OperationConfig, TrainerConfig
from .config_utils import ProfilerConfig, UtilsConfig

load_dotenv()


def _sanitize_path_component(path_component: str) -> str:
    """
    Sanitize a path component to prevent directory traversal attacks.

    Removes any directory components and only keeps the base filename.
    """
    return os.path.basename(path_component)


def _validate_path_within_dir(path: Path, base_dir: Path) -> bool:
    """
    Validate that a path resolves to a location within the base directory.

    This prevents path traversal attacks where malicious input like
    '../../etc/passwd' could be used to access files outside the intended directory.
    """
    try:
        resolved_path = path.resolve()
        resolved_base = base_dir.resolve()
        return str(resolved_path).startswith(str(resolved_base))
    except (OSError, ValueError):
        return False


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


def _config_validation(config: MainConfig):
    """Validate arguments and update internal enum if necessary"""
    # train steps
    if config.operation.train_steps <= 0:
        raise ValueError("operation.train_steps should be larger than 0")

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


# arguments utilities
def parse_args():
    """Parse command line arguments."""

    parser = ArgumentParser(prog="trainer configuration", description="LLM trainer")

    # configuration arguments - use prefixed names to avoid collisions
    for group_field in fields(MainConfig):
        group_name = group_field.name
        for field_ in fields(group_field.type):
            # Prefix argument with group name (e.g., --model.name, --trainer.batch_size)
            arg_name = f"--{group_name}.{field_.name}"
            parser.add_argument(arg_name, **field_.metadata)

    parser.add_argument("--config-path", type=str, default=None, help="yaml config file path")
    parser.add_argument(
        "--local-rank",
        dest="local_rank",
        default=0,
        type=int,
        help="local rank for ddp distributed training",
    )

    # parse argument inputs
    args = parser.parse_args()
    return args


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
            else:
                base_dir = Path(os.environ["DATASET_DIR"]) / "preprocssed_corpus"

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


def _update_config_from_args(config: dataclass, args):
    """update config from command line using recursive dot-notation."""

    arg_dict = vars(args)

    def set_recursive_attr(obj, attr_path, value):
        parts = attr_path.split(".")
        for part in parts[:-1]:
            obj = getattr(obj, part)

        target_attr = parts[-1]

        # Find the field type for casting
        field_info = [f for f in fields(obj) if f.name == target_attr]
        if not field_info:
            raise AttributeError(f"Attribute {target_attr} not found")

        field_type = field_info[0].type

        # Handle Optional/Union types
        if get_origin(field_type) is Optional:
            type_ = get_args(field_type)[0]
        elif get_origin(field_type) is Union:
            # Simple heuristic for common types
            for type_cls in [int, float, str, list, bool]:
                if isinstance(value, type_cls):
                    type_ = type_cls
                    break
        else:
            type_ = field_type

        # Handle Boolean strings from argparse
        if type_ is bool and isinstance(value, str):
            value = value.lower() in ("true", "1", "yes")

        setattr(obj, target_attr, type_(value))

    for arg_name, arg_value in arg_dict.items():
        if arg_value is None or arg_name in ["config_path", "local_rank"]:
            continue
        try:
            # The attribute path from argparse already contains the group,
            # so we should start from the top-level `config` object.
            set_recursive_attr(config, arg_name, arg_value)
        except (AttributeError, IndexError):
            # This can happen if an argument from argparse doesn't map to a config path.
            # We can either warn or ignore. For now, we'll ignore to match the
            # previous behavior of trying the next group.
            continue
        except Exception as e:
            raise ValueError(f"Error processing argument '{arg_name}': {e}")


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
        if isinstance(sub_group_config, str):
            # add config name to config group
            getattr(config, config_group_key).name = sub_group_config

            # Sanitize path component to prevent directory traversal
            sanitized_config_name = _sanitize_path_component(sub_group_config)

            # load sub-config: data, model config
            # Resolve sub-config path relative to the project's 'configs/' directory
            config_base_dir = Path(args.config_path).parent
            sub_group_config_path = (
                config_base_dir / f"{config_group_key}/{sanitized_config_name}.yaml"
            )

            # Validate path is within expected directory
            if not _validate_path_within_dir(sub_group_config_path, config_base_dir):
                raise ValueError(
                    f"Invalid config path: {sub_group_config} resolves outside the config directory"
                )

            if sanitized_config_name == "dummy":
                # load dummy config if it exists or run with default dummy config
                # this is usually for dummy model usage
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


def load_trainer_config() -> MainConfig:
    """config trainer's arguments from command line and config file."""

    config = MainConfig(
        model=ModelConfig(),
        init=InitConfig(),
        optim=OptimConfig(),
        data=DataConfig(),
        parallel=ParallelConfig(),
        trainer=TrainerConfig(),
        operation=OperationConfig(),
        utils=UtilsConfig(),
        profiler=ProfilerConfig(),
        peft=PEFTConfig(),
        alignment=AlignmentConfig(),
    )

    # get config from command line
    args = parse_args()

    # get config from yaml config file
    if hasattr(args, "config_path") and args.config_path is not None:
        _load_config_from_yaml(config, args)

    # update config from command line arguments
    _update_config_from_args(config, args)

    # Args from environment
    config.parallel.rank = int(os.getenv("RANK", "0"))
    config.parallel.local_rank = int(os.getenv("LOCAL_RANK", "0"))
    config.parallel.world_size = int(os.getenv("WORLD_SIZE", "1"))

    # load special tokens
    if config.trainer.special_tokens_config_path:
        base_dir = (
            config.model.vocab_name_or_path
            if config.model.vocab_name_or_path
            else config.trainer.model_path
        )
        base_dir_path = Path(base_dir)

        # Sanitize path to prevent directory traversal
        sanitized_path = _sanitize_path_component(config.trainer.special_tokens_config_path)
        special_token_file_path = base_dir_path / sanitized_path

        # Validate path is within base directory
        if not _validate_path_within_dir(special_token_file_path, base_dir_path):
            raise ValueError("Invalid special tokens path: resolves outside the base directory")

        if special_token_file_path.exists():
            with open(special_token_file_path, encoding="utf-8") as f:
                import json

                config.trainer.special_tokens_config = json.load(f)
        else:
            raise FileNotFoundError(
                f"Could not find special token config file: {special_token_file_path}"
            )
    delattr(config.trainer, "special_tokens_config_path")

    _config_validation(config)

    return config


def print_args(config):
    """Print config."""
    for group_field in fields(config):
        for field_ in fields(group_field.type):
            print(f"{field_.name}: {getattr(config, field_.name)}")
