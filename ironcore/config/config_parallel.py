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
from typing import Literal, Optional

from .config import BaseConfig


@dataclass
class ParallelConfig(BaseConfig):
    rank: int = field(default=-1, metadata={"help": "global rank"})
    local_rank: int = field(default=0, metadata={"help": "local rank"})
    world_size: int = field(default=1, metadata={"help": "world size"})
    dist_backend: str = field(default="nccl", metadata={
                              "help": "distributed backend"})
    timeout_minute: float = field(
        default=10.0, metadata={"help": "distributed timeout in minutes"}
    )

    use_fsdp: bool = field(default=False, metadata={"help": "use FSDP"})
    fsdp_offload_params: bool = field(
        default=False, metadata={"help": "FSDP cpu offload"}
    )
    fsdp_mixed_precision: Literal[
        "fp16", "float16", "bf16", "bfloat16", "fp32", "float32", "mixed"
    ] = field(
        default="mixed",
        metadata={
            "help": f'FSDP mixed precision mode: {"mixed", "fp16", "float16", "bf16", "bfloat16", "fp32", "float32"}'
        },
    )
    fsdp_sharding_strategy: Literal["full", "hybrid", "no_shard"] = field(
        default="full",
        metadata={
            "help": f'FSDP sharding strategy: {"full", "hybrid", "no_shard"}'},
    )
    fsdp_state_dict_type: Literal["full", "local", "sharded"] = field(
        default="full",
        metadata={"help": f'FSDP state dict type: {"full", "local", "sharded"}'},
    )
