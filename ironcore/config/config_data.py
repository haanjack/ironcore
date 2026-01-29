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
from typing import List, Optional, Union

from ironcore.config import BaseConfig


@dataclass
class DataConfig(BaseConfig):
    """config for data."""

    task_type: str = field(
        default="pretrain",
        metadata={"help": "Training task type: pretrain, sft, dpo"},
    )
    data_path: Optional[List[Union[float, str]]] = field(
        default_factory=lambda: None,
        metadata={"help": "base data path with multiple datasets"},
    )
    splits: List[float] = field(
        default_factory=lambda: [0.97, 0.029, 0.001],
        metadata={"help": "train/eval/test split ratios"},
    )
    train_datasets: Optional[List[Union[float, str]]] = field(
        default_factory=lambda: None, metadata={"help": "train datasets"}
    )
    eval_datasets: Optional[List[Union[float, str]]] = field(
        default_factory=lambda: None, metadata={"help": "eval datasets"}
    )
    test_datasets: Optional[List[Union[float, str]]] = field(
        default_factory=lambda: None, metadata={"help": "test datasets"}
    )

    # embedding
    vocab_size: int = field(default=51200, metadata={"help": "vocab size"})
    num_token_types: int = field(
        default=2, metadata={"help": "number of token types"})

    # data loading control
    pad_to_max_length: bool = field(
        default=False,
        metadata={
            "help": "Pad rest of sample when input sequence length is shorter than max sequence length in training"
        },
    )
    pad_token_id: int = field(
        default=-1,
        metadata={
            "help": "Input sample's PAD token id. By default, it is EOS token."},
    )
