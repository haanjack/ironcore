# Copyright (c) 2025-2026 Jaegeun Han
#
# SPDX-License-Identifier: MIT
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the above copyright notice,
# this list of conditions, and the following disclaimer are retained.
#
# Full license text is available at LICENSE file.

import os
from argparse import ArgumentParser, Namespace
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, get_args, get_origin

import torch
from dotenv import load_dotenv

from ironcore.utils import get_dataset_base_dir, load_yaml_config

from .config import BaseConfig
from .config_data import DataConfig
from .config_model import ModelConfig
from .config_optim import OptimConfig
from .config_parallel import ParallelConfig
from .config_trainer import InitConfig, OperationConfig, TrainerConfig
from .config_utils import UtilsConfig

load_dotenv()


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


def _config_validation(config: MainConfig):
    """Validate arguments and update internal enum if necessary"""
    # train steps
    if config.operation.train_steps <= 0:
        raise ValueError("operation.train_steps should be larger than 0")

    dp_group_size = (
        config.trainer.tensor_model_parallel_size
        * config.trainer.pipeline_model_parallel_size
    )
    dp_world_size = config.parallel.world_size // dp_group_size
    assert (
        dp_world_size > 0
    ), f"World size ({config.parallel.world_size}) is smaller than single data parallelism group size ({dp_group_size})"

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
                config.trainer.train_batch_size
                // config.trainer.micro_batch_size
                // dp_world_size
            )
        elif config.trainer.micro_batch_size is None:
            config.trainer.micro_batch_size = (
                config.trainer.train_batch_size
                // config.trainer.gradient_accumulation_steps
                // dp_world_size
            )

        assert config.trainer.train_batch_size % (config.trainer.micro_batch_size * dp_world_size) == 0, (
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
        config.trainer.micro_batch_size
        * config.trainer.gradient_accumulation_steps
        * dp_world_size
        != config.trainer.train_batch_size
    ):
        raise ValueError(
            "micro_batch_size * gradient_accumulation_steps should be equal to train_batch_size"
        )

    # model parallel validation
    if config.trainer.tensor_model_parallel_size > 1 and config.model.name != "dummy":
        if (
            config.model.num_attention_heads % config.trainer.tensor_model_parallel_size
            != 0
        ):
            raise ValueError(
                "num_attention_heads should be divisible by tensor_model_parallel_size"
            )
        if (
            config.model.num_attention_groups
            % config.trainer.tensor_model_parallel_size
            != 0
        ):
            raise ValueError(
                "num_attention_groups should be divisible by tensor_model_parallel_size"
            )

    # positional embedding
    if not config.model.positional_embedding.type.lower() in [
        "absolute",
        "rope",
        "none",
    ]:
        raise ValueError(
            "Available positional embeddong options are ['absolute', 'rope', 'none']."
        )

    if not torch.cuda.is_available() and config.trainer.tensor_model_parallel_size > 1:
        raise ValueError(
            "tensor_model_parallel_size should be 1 in non-CUDA environments"
        )


# arguments utilities
def parse_args():
    """Parse command line arguments."""

    parser = ArgumentParser(prog="trainer configuration",
                            description="LLM trainer")

    # configuration arguments
    configs = [config.type for config in fields(MainConfig)]
    for category in configs:
        for field_ in fields(category):
            parser.add_argument(f"--{field_.name}", **field_.metadata)

    parser.add_argument(
        "--config-path", type=str, default=None, help="yaml config file path"
    )
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


def load_data_config(config, datasets: Dict[str, Any]) -> List[Dict[str, Any]]:
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
                base_dir = Path(
                    os.environ["DATASET_DIR"]) / "preprocssed_corpus"

            dataset_name = dataset_name_or_path
            if dataset_config.get("subgroup", None):
                dataset_name = f"{dataset_name}_{dataset_config.get('subgroup')}"
            loaded_config["name"] = dataset_name
            loaded_config["dataset_path"] = base_dir / \
                config.data.name / dataset_name

        if "ratio" in dataset_config:
            loaded_config["ratio"] = dataset_config.get("ratio", 1.0)
        if "samples" in dataset_config:
            loaded_config["samples"] = dataset_config.get("samples")

        output_list.append(loaded_config)

    return output_list


# data config
def _update_config_from_yaml(
    config: dataclass, config_group_key: str, config_group: dict
):
    """update config from yaml config file."""

    # get config from yaml
    config_dict_item = asdict(config)[config_group_key]
    for yaml_config_key, yaml_config_value in config_group.items():
        assert (
            yaml_config_key in config_dict_item
        ), f"{yaml_config_key} is not defined in {config_group_key}. Check yaml config file."
        config_dict_item[yaml_config_key] = yaml_config_value

    # update config
    getattr(config, config_group_key)(**config_dict_item)


def _update_data_config_from_yaml(
    config: dataclass, config_group_key, config_group: dict
):
    """update data config from yaml config file."""

    for sub_group_key, sub_group_value in config_group.items():
        if sub_group_key in ["train", "eval", "test"]:
            sub_group_key = f"{sub_group_key}_datasets"
            # config.data.__dict__[sub_group_key
            data_group_config = load_data_config(config, sub_group_value)
            setattr(config.data, sub_group_key, data_group_config)
        else:
            assert (
                sub_group_key in config_group
            ), f"{sub_group_key} is not defined in {config_group_key}. Check yaml config file."
            # config.data.__dict__[sub_group_key] = sub_group_value
            setattr(config.data, sub_group_key, sub_group_value)


def _load_config_from_yaml(config: dataclass, args):
    """
    load config from yaml config file.
    """
    yaml_config = load_yaml_config(args.config_path)
    config_dict = asdict(config)
    # load configs from yaml
    for yaml_config_group_key, yaml_config_group in yaml_config.items():
        # check if config group is defined
        assert (
            yaml_config_group_key in config_dict
        ), f"{yaml_config_group_key} is not defined configuration group"

        if yaml_config_group is None:
            getattr(config, yaml_config_group_key).attr_name = "dummy"

        # load configs from subsidary yaml config files
        if isinstance(yaml_config_group, str):
            # load sub-config defined in seperated file: model and data config
            sub_group_config_path = (
                Path(args.config_path).parent
                / f"{yaml_config_group_key}/{yaml_config_group}.yaml"
            )
            sub_group_config = load_yaml_config(sub_group_config_path)

            if yaml_config_group_key == "data":
                _update_data_config_from_yaml(
                    config, yaml_config_group_key, sub_group_config
                )
            else:
                _update_config_from_yaml(
                    config, yaml_config_group_key, sub_group_config
                )
                getattr(config, yaml_config_group_key).attr_name = yaml_config_group
        else:
            # load configs: trainer, optimizer, etc
            _update_config_from_yaml(
                config, yaml_config_group_key, yaml_config_group)


def _update_config_from_args(config: dataclass, args):
    """update config from command line."""
    for group_field in fields(config):
        for field_ in fields(group_field.type):
            # get argument and update config if it is defined argument
            if not hasattr(args, field_.name) or getattr(args, field_.name) is None:
                continue

            if field_.name == "config_path":
                # skip config path argument
                continue

            config_group = getattr(config, group_field.name)

            # load optional type as config defined type_
            if get_origin(field_.type) is Optional:
                type_ = get_args(field_.type)[0]
            elif get_origin(field_.type) is Union:
                for type_cls in [int, float, str, list]:
                    if isinstance(getattr(args, field_.name), type_cls):
                        type_ = type_cls
                        break
            else:
                type_ = field_.type

            value = type_(getattr(args, field_.name))
            setattr(config_group, field_.name, value)


def _update_config_from_yaml(
    config: dataclass, config_group_key: str, config_group: dict
):
    """update config from yaml config file."""

    # get config from yaml
    config_dict_item = asdict(config)[config_group_key]
    for yaml_config_key, yaml_config_value in config_group.items():
        assert (
            yaml_config_key in config_dict_item
        ), f"{yaml_config_key} is not defined in {config_group_key} config. Check yaml config file."
        config_dict_item[yaml_config_key] = yaml_config_value

    # update config
    getattr(config, config_group_key)(**config_dict_item)


def _update_data_config_from_yaml(
    config: dataclass, config_group_key, config_group: dict
):
    """update data config from yaml config file."""

    for sub_group_key, sub_group_value in config_group.items():
        if sub_group_key in ["train", "eval", "test"]:
            sub_group_key = f"{sub_group_key}_datasets"
            config.data.__dict__[sub_group_key] = load_data_config(
                config, sub_group_value
            )
        else:
            # update arguments for DataConfig class
            assert (
                sub_group_key in config_group
            ), f"{sub_group_key} is not defined in {config_group_key}. Check yaml config file."
            config.data.__dict__[sub_group_key] = sub_group_value


def _load_subgroup_config_from_yaml(config, config_group_key, sub_group_config):
    """
    Load subgroup config
    """
    if config_group_key == "data":
        _update_data_config_from_yaml(
            config, config_group_key, sub_group_config)
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
        assert (
            config_group_key in config_dict
        ), f"{config_group_key} is not defined configuration group"

        # load configs from yaml
        if isinstance(sub_group_config, str):
            # add config name to config group
            getattr(config, config_group_key).name = sub_group_config

            # load sub-config: data, model config
            sub_group_config_path = (
                Path(args.config_path).parent
                / f"{config_group_key}/{sub_group_config}.yaml"
            )
            if sub_group_config in "dummy":
                # load dummy config if it exists or run with default dummy config
                # this is usually for dummy model usage
                if sub_group_config_path.exists():
                    sub_group_config_from_file = load_yaml_config(
                        sub_group_config_path)
                continue
            else:
                if not sub_group_config_path.exists():
                    raise FileNotFoundError(
                        f"Config file not found: {sub_group_config_path}"
                    )
                sub_group_config_from_file = load_yaml_config(
                    sub_group_config_path)

            _load_subgroup_config_from_yaml(
                config, config_group_key, sub_group_config_from_file
            )
        else:
            _load_subgroup_config_from_yaml(
                config, config_group_key, sub_group_config)


def _update_config_from_args(config: dataclass, args):
    """update config from command line."""
    for group_field in fields(config):
        for field_ in fields(group_field.type):
            # get argument and update config if it is defined argument
            if not hasattr(args, field_.name) or getattr(args, field_.name) is None:
                continue

            if field_.name == "config_path":
                # skip config path argument
                continue

            config_group = getattr(config, group_field.name)

            # load optional type as config defined type_
            if get_origin(field_.type) is Optional:
                type_ = get_args(field_.type)[0]
            elif get_origin(field_.type) is Union:
                for type_cls in [int, float, str, list]:
                    if isinstance(getattr(args, field_.name), type_cls):
                        type_ = type_cls
                        break
            else:
                type_ = field_.type

            value = type_(getattr(args, field_.name))
            setattr(config_group, field_.name, value)


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
        special_token_file_path = (
            Path(base_dir) / config.trainer.special_tokens_config_path
        )
        if special_token_file_path.exists():
            with open(special_token_file_path, "r", encoding="utf-8") as f:
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
