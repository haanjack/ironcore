# Copyright (c) 2025-2026 Jaegeun Han
#
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass, field
from typing import Literal

from .config import BaseConfig


@dataclass
class TrainerConfig(BaseConfig):
    model_name: str = field(default="model", metadata={"help": "model name"})
    config_path: str | None = field(default=None, metadata={"help": "config file path"})
    model_path: str = field(default="", metadata={"help": "model save/load dir"})
    load_from_hf: str | None = field(
        default=None,
        metadata={
            "help": "HuggingFace model name/path to load pretrained weights (e.g., 'Qwen/Qwen2.5-0.5B-Instruct')"
        },
    )

    micro_batch_size: int | None = field(default=2, metadata={"help": "micro batch size"})
    train_batch_size: int | None = field(default=None, metadata={"help": "batch size"})
    gradient_accumulation_steps: int | None = field(
        default=None, metadata={"help": "gradient accumulation steps"}
    )

    # evaluation
    do_eval: bool = field(default=False, metadata={"help": "do evaluation"})
    eval_batch_size: int | None = field(default=None, metadata={"help": "eval batch size"})
    do_eval_subtask: bool = field(
        default=False, metadata={"help": "do evaluation using specified subtasks"}
    )

    test_batch_size: int | None = field(default=None, metadata={"help": "test batch size"})
    do_test: bool = field(default=False, metadata={"help": "do prediction"})

    save_checkpoint_steps: int = field(default=1000, metadata={"help": "save checkpoint steps"})
    log_interval: int = field(default=20, metadata={"help": "print progress steps"})

    # Norm computation cadence
    # Allowed: None, 'log', 'checkpoint'
    # None disables the specific norm computation; 'log' computes on log_interval; 'checkpoint' computes on save_checkpoint_steps
    grad_norm_log_interval: Literal["log", "checkpoint"] | None = field(
        default=None,
        metadata={"help": "gradient norm cadence: None | 'log' | 'checkpoint'"},
    )
    param_norm_log_interval: Literal["log", "checkpoint"] | None = field(
        default=None,
        metadata={"help": "parameter norm cadence: None | 'log' | 'checkpoint'"},
    )

    num_workers: int = field(
        default=8, metadata={"help": "number of workers in dataset processing"}
    )

    # model parallelism
    tensor_model_parallel_size: int = field(
        default=1, metadata={"help": "model parallel size on a same transformer layer"}
    )

    vocab_padding_unit: int = field(
        default=128,
        metadata={"help": "vocab padding unit for tensor core optimization"},
    )

    # special tokens
    special_tokens_config_path: str | None = field(
        default=None, metadata={"help": "special token file path"}
    )

    use_flash_attn: bool = field(
        default=True, metadata={"help": "use flash attention for the attention layer"}
    )

    # KV Cache for evaluation
    use_kv_cache_in_eval: bool = field(
        default=True,
        metadata={"help": "Use KV cache during evaluation for faster inference"},
    )

    # Async Tensor Parallelism
    sequence_chunk_size: int | None = field(
        default=None,
        metadata={
            "help": "Target chunk size (in tokens) for async tensor parallelism. If set, sequence is split into chunks of this size."
        },
    )

    # torch.compile options
    compile_model: bool = field(
        default=False, metadata={"help": "Enable torch.compile for the model"}
    )
    compile_mode: Literal["default", "reduce-overhead", "max-autotune"] | None = field(
        default="default",
        metadata={"help": "torch.compile mode: default | reduce-overhead | max-autotune"},
    )
    compile_backend: Literal["inductor", "cudagraphs", "eager"] = field(
        default="inductor",
        metadata={"help": "torch.compile backend: inductor | cudagraphs | eager"},
    )
    compile_dynamic: bool = field(
        default=False,
        metadata={
            "help": "Enable dynamic shapes to avoid recompilation on shape changes (e.g., train/eval batch size differences)"
        },
    )
    compile_fullgraph: bool = field(
        default=False,
        metadata={
            "help": "Require full graph compilation without graph breaks. Fails if graph breaks are unavoidable."
        },
    )


@dataclass
class OperationConfig(BaseConfig):
    """config for trainer's operation"""

    train_steps: int = field(default=1000, metadata={"help": "train steps"})
    eval_interval: int = field(default=100, metadata={"help": "evaluation interval"})

    # TODO: deprecates and put samples can be set individually
    eval_samples: int = field(default=100, metadata={"help": "evaluation sample size"})
    test_samples: int = field(default=100, metadata={"help": "test sample size"})
    activation_recompute: bool = field(
        default=False,
        metadata={"help": "use activation recompute to reduce memory footprints in training"},
    )
    recompute_strategy: str | None = field(
        default="default",
        metadata={
            "help": "Choose PyTorch's recompute activation strategy (default: use_reentrant=False, optimized: use_reentrant=True)"
        },
    )

    no_save: bool = field(default=False, metadata={"help": "no save checkpoint"})
    exit_interval: int | None = field(default=None, metadata={"help": "exit interval"})
    save_dist_ckpt: bool = field(
        default=False, metadata={"help": "use distributed save checkpoint"}
    )
    save_full_model: bool = field(
        default=False,
        metadata={
            "help": "When PEFT is active, save the full model (base + adapter). "
            "Default (False) saves adapter weights only — much smaller and "
            "distributable as a standalone adapter. (Fable issue #65.)"
        },
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
    init_std: float = field(default=0.006, metadata={"help": "initialization std"})
    xavier_init: bool | None = field(
        default=False, metadata={"help": "Use Xavier initialization method"}
    )
