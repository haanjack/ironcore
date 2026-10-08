# Copyright (c) 2025-2026 Jaegeun Han
# SPDX-License-Identifier: Apache-2.0

"""Execution contract for checkpointed token-block MLPs."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from . import MainConfig


def validate_blockwise_mlp(config: MainConfig) -> None:
    """Reject unsupported combinations before allocating training resources."""
    size = config.trainer.mlp_chunk_size
    grouped = config.model.moe.use_moe and config.model.moe.expert_backend == "grouped"
    if size is None and not grouped:
        return
    if size is not None and (not isinstance(size, int) or isinstance(size, bool) or size < 1):
        raise ValueError("mlp_chunk_size must be a positive integer or None")
    if grouped and config.model.moe.expert_model_parallel_size != 1:
        raise ValueError("Grouped expert backend currently requires EP=1")
    if config.model.is_gemma4:
        raise ValueError("Block-wise MLP currently supports generic dense and MoE decoders")
    if config.model.dropout_mlp or (config.peft.method == "lora" and config.peft.lora.dropout):
        raise ValueError("Block-wise MLP currently requires zero MLP/LoRA dropout")
    if config.offload.enabled or config.parallel.use_fsdp:
        raise ValueError("Block-wise MLP offload and FSDP have not been validated")
