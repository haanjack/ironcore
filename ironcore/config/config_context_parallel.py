# Copyright (c) 2025-2026 Jaegeun Han
# SPDX-License-Identifier: Apache-2.0

"""Supported execution contract for causal dense context parallel training."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from . import MainConfig


def validate_context_parallel(config: MainConfig) -> None:
    """Reject combinations whose token/loss/communication semantics are unsupported."""
    size = config.trainer.context_parallel_size
    if not isinstance(size, int) or isinstance(size, bool) or size < 1:
        raise ValueError("context_parallel_size must be a positive integer")
    if config.trainer.context_parallel_backend not in {"ring", "sdpa"}:
        raise ValueError("context_parallel_backend must be ring or sdpa")
    if size == 1:
        return
    tp_size = config.trainer.tensor_model_parallel_size
    if not isinstance(tp_size, int) or isinstance(tp_size, bool) or tp_size < 1:
        raise ValueError("tensor_model_parallel_size must be a positive integer")
    if config.parallel.world_size % (size * tp_size):
        raise ValueError("world_size must be divisible by TP * CP")
    if config.trainer.context_parallel_backend == "ring":
        if config.model.precision.lower() not in {"float16", "fp16", "bfloat16", "bf16"}:
            raise ValueError("CP ring requires CUDA FP16/BF16 compute; use sdpa for FP32")
        if not 8 <= config.model.head_dim <= 256 or config.model.head_dim % 8:
            raise ValueError("CP ring FlashAttention requires head_dim divisible by 8 in [8, 256]")
    if config.model.is_gemma4:
        gemma = config.model.gemma4
        if config.trainer.context_parallel_backend != "sdpa" or gemma.attention_chunk_size is None:
            raise ValueError(
                "Gemma 4 CP requires sdpa with query-block attention; 512-dimensional global heads cannot use the ring FlashAttention kernel"
            )
        if gemma.hidden_size_per_layer_input or gemma.num_kv_shared_layers:
            raise ValueError("Gemma 4 CP currently requires no PLE or shared KV")
    if config.model.moe.use_moe:
        ep_size = config.model.moe.expert_model_parallel_size
        if ep_size > 1 and (
            ep_size != 2 or tp_size != 1 or config.parallel.world_size != ep_size * size
        ):
            raise ValueError("Context parallel MoE EP requires EP=2, TP=1, world=2*CP")
        if config.model.moe.router_jitter_noise:
            raise ValueError("Context parallel MoE currently requires zero router jitter")
    if (
        config.parallel.use_fsdp
        or config.parallel.use_distributed_optimizer
        or (config.offload.enabled and not config.model.is_gemma4)
    ):
        raise ValueError(
            "Context parallel FSDP, distributed optimizer and offload are not supported"
        )
    unpacked_sft = (
        config.model.is_gemma4 and config.data.task_type == "sft" and not config.data.sft_packing
    )
    if config.data.task_type != "pretrain" and not unpacked_sft:
        raise ValueError("Context parallel currently requires causal pretrain token-mean loss")
    if config.model.reset_attention_mask or config.model.reset_position_ids:
        raise ValueError("Context parallel does not support packed/reset document attention")
    if config.model.kv_cache.use_paged:
        raise ValueError("Context parallel does not support paged KV caches")
    dropout = [config.model.dropout_attn, config.model.dropout_mlp, config.model.dropout_embd]
    if config.peft.method == "lora":
        dropout.append(config.peft.lora.dropout)
    if any(dropout):
        raise ValueError(
            "Context parallel currently requires zero dropout for reproducible recomputation"
        )
    if config.init.data_parallel_random_init:
        raise ValueError(
            "Context parallel requires identical parameter initialization across replicas"
        )
