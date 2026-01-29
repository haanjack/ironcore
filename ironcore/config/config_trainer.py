# Copyright (c) 2025-2026 Jaegeun Han
#
# SPDX-License-Identifier: MIT
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the above copyright notice,
# this list of conditions, and the following disclaimer are retained.
#
# Full license text is available at LICENSE file.

from dataclasses import dataclass, field
from typing import Optional, Literal

from .config import BaseConfig


@dataclass
class TrainerConfig(BaseConfig):
    model_name: str = field(default="model", metadata={"help": "model name"})
    config_path: Optional[str] = field(
        default=None, metadata={"help": "config file path"}
    )
    model_path: str = field(
        default="", metadata={"help": "model save/load dir"}
    )

    micro_batch_size: Optional[int] = field(
        default=2, metadata={"help": "micro batch size"}
    )
    train_batch_size: Optional[int] = field(
        default=None, metadata={"help": "batch size"}
    )
    gradient_accumulation_steps: Optional[int] = field(
        default=None, metadata={"help": "gradient accumulation steps"}
    )

    # evaluation
    do_eval: bool = field(default=False, metadata={"help": "do evaluation"})
    eval_batch_size: Optional[int] = field(
        default=None, metadata={"help": "eval batch size"}
    )
    do_eval_subtask: bool = field(
        default=False, metadata={"help": "do evaluation using specified subtasks"}
    )

    test_batch_size: Optional[int] = field(
        default=None, metadata={"help": "test batch size"}
    )
    do_test: bool = field(default=False, metadata={"help": "do prediction"})

    save_checkpoint_steps: int = field(
        default=1000, metadata={"help": "save checkpoint steps"}
    )
    log_interval: int = field(
        default=20, metadata={"help": "print progress steps"}
    )

    # Norm computation cadence
    # Allowed: None, 'log', 'checkpoint'
    # None disables the specific norm computation; 'log' computes on log_interval; 'checkpoint' computes on save_checkpoint_steps
    grad_norm_log_interval: Optional[Literal['log', 'checkpoint']] = field(
        default=None,
        metadata={
            "help": "gradient norm cadence: None | 'log' | 'checkpoint'"
        },
    )
    param_norm_log_interval: Optional[Literal['log', 'checkpoint']] = field(
        default=None,
        metadata={
            "help": "parameter norm cadence: None | 'log' | 'checkpoint'"
        },
    )

    num_workers: int = field(
        default=8, metadata={"help": "number of workers in dataset processing"}
    )

    # model parallelism
    tensor_model_parallel_size: int = field(
        default=1, metadata={"help": "model parallel size on a same transformer layer"}
    )
    pipeline_model_parallel_size: int = field(
        default=1,
        metadata={"help": "pipeline model parallel size splits model vertically"},
    )

    vocab_padding_unit: int = field(
        default=128,
        metadata={"help": "vocab padding unit for tensor core optimization"},
    )

    # special tokens
    special_tokens_config_path: Optional[str] = field(
        default=None, metadata={"help": "special token file path"}
    )

    use_flash_attn: bool = field(
        default=True, metadata={"help": "use flash attention for the attention layer"}
    )

    # torch.compile options
    compile_model: bool = field(
        default=False, metadata={"help": "Enable torch.compile for the model"}
    )
    compile_mode: Optional[Literal["default", "reduce-overhead", "max-autotune"]] = field(
        default="default",
        metadata={"help": "torch.compile mode: default | reduce-overhead | max-autotune"}
    )
    compile_backend: Literal["inductor", "cudagraphs", "eager"] = field(
        default="inductor",
        metadata={"help": "torch.compile backend: inductor | cudagraphs | eager"}
    )
    compile_dynamic: bool = field(
        default=False,
        metadata={"help": "Enable dynamic shapes to avoid recompilation on shape changes (e.g., train/eval batch size differences)"}
    )
    compile_fullgraph: bool = field(
        default=False,
        metadata={"help": "Require full graph compilation without graph breaks. Fails if graph breaks are unavoidable."}
    )



@dataclass
class OperationConfig(BaseConfig):
    """config for trainer's operation"""

    train_steps: int = field(default=1000, metadata={"help": "train steps"})
    eval_interval: int = field(default=100, metadata={
                               "help": "evaluation interval"})

    # TODO: deprecates and put samples can be set individually
    eval_samples: int = field(default=100, metadata={
                              "help": "evaluation sample size"})
    test_samples: int = field(default=100, metadata={
                              "help": "test sample size"})
    activation_recompute: bool = field(
        default=False,
        metadata={
            "help": "use activation recompute to reduce memory footprints in training"
        }
    )
    recompute_strategy: Optional[str] = field(
        default="default",
        metadata={
            "help": "Choose PyTorch's recompute activation strategy (default: use_reentrant=False, optimized: use_reentrant=True)"
        },
    )

    no_save: bool = field(default=False, metadata={
                          "help": "no save checkpoint"})
    exit_interval: Optional[int] = field(
        default=None, metadata={"help": "exit interval"}
    )
    save_dist_ckpt: bool = field(
        default=False, metadata={"help": "use distributed save checkpoint"}
    )


@dataclass
class InitConfig(BaseConfig):
    """model weight initialization config"""

    seed: int = field(
        default=1337,
        metadata={"help": "Random seed for python, numpy, pytorch, and cuda"},
    )
    data_parallel_random_init: bool = field(
        default=False, metadata={"help": "Enable data parallel random init"}
    )
    init_std: float = field(default=0.006, metadata={
                            "help": "initialization std"})
    xavier_init: Optional[bool] = field(
        default=False, metadata={"help": "Use Xavier initialization method"}
    )
