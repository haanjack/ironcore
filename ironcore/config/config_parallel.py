# Copyright (c) 2025-2026 Jaegeun Han
#
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass, field
from typing import Literal

from .config import BaseConfig


@dataclass
class ParallelConfig(BaseConfig):
    rank: int = field(default=-1, metadata={"help": "global rank"})
    local_rank: int = field(default=0, metadata={"help": "local rank"})
    world_size: int = field(default=1, metadata={"help": "world size"})
    dist_backend: str = field(default="nccl", metadata={"help": "distributed backend"})
    timeout_minute: float = field(default=10.0, metadata={"help": "distributed timeout in minutes"})

    use_fsdp: bool = field(default=False, metadata={"help": "use FSDP"})
    fsdp_offload_params: bool = field(default=False, metadata={"help": "FSDP cpu offload"})
    fsdp_pin_memory: bool = field(
        default=True,
        metadata={"help": "Use pinned memory for faster CPU-GPU transfer when offloading"},
    )
    fsdp_use_orig_params: bool = field(
        default=False,
        metadata={
            "help": "Use original parameters (better optimizer compatibility, e.g., for torch.compile)"
        },
    )
    fsdp_mixed_precision: Literal[
        "fp16", "float16", "bf16", "bfloat16", "fp32", "float32", "mixed"
    ] = field(
        default="mixed",
        metadata={
            "help": f"FSDP mixed precision mode: {'mixed', 'fp16', 'float16', 'bf16', 'bfloat16', 'fp32', 'float32'}"
        },
    )
    fsdp_sharding_strategy: Literal["full", "hybrid", "no_shard", "shard_grad_op"] = field(
        default="full",
        metadata={
            "help": "FSDP sharding strategy: "
            "'full' - shard params, grads, optimizer states; "
            "'hybrid' - hybrid sharding for multi-node; "
            "'no_shard' - no sharding (like DDP); "
            "'shard_grad_op' - only shard grads and optimizer states (faster with CPU offload)"
        },
    )
    fsdp_state_dict_type: Literal["full", "local", "sharded"] = field(
        default="full",
        metadata={"help": f"FSDP state dict type: {'full', 'local', 'sharded'}"},
    )
