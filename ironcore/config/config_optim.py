# Copyright (c) 2025-2026 Jaegeun Han
#
# SPDX-License-Identifier: MIT
#
# Redistribution and use in source and binary form1s, with or without
# modification, are permitted provided that the above copyright notice,
# this list of conditions, and the following disclaimer are retained.
#
# Full license text is available at LICENSE file.

from dataclasses import dataclass, field

from .config import BaseConfig


@dataclass
class OptimConfig(BaseConfig):
    """config for optimizer."""

    optimizer: str = field(default="adam", metadata={"help": "optimizer"})

    lr_scheduler: str = field(
        default="cosine", metadata={"help": "learning rate scheduler"}
    )

    max_lr: float = field(default=5e-4, metadata={"help": "learning rate"})
    min_lr: float = field(default=0.0, metadata={
                          "help": "minimum learning rate"})

    warmup_steps: int = field(
        default=0, metadata={"help": "total number of lr warmup steps"}
    )
    annealing_steps: int = field(
        default=0, metadata={"help": "total number of lr annealing steps"}
    )

    weight_decay: float = field(default=0.01, metadata={"help": "weight decay"})
    no_decay_on_embedding: bool = field(
        default=True,
        metadata={"help": "no weight decay on embedding layer"},
    )
    adam_eps: float = field(default=1e-8, metadata={"help": "adam epsilon"})
    adam_beta1: float = field(default=0.9, metadata={"help": "adam beta1"})
    adam_beta2: float = field(default=0.95, metadata={"help": "adam beta2"})
    clip_grad: float = field(
        default=1.0, metadata={"help": "gradient clipping scale value"}
    )

    # checkpointing option
    load_checkpoint_optim_state: bool = field(
        default=True,
        metadata={
            "help": "Whether to load and use optimizer states from the checkpoint"
        },
    )
    load_checkpoint_lr_scheduler: bool = field(
        default=True,
        metadata={
            "help": "Whether to load and use lr scheduler info from the checkpoint"
        },
    )
