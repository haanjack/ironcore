# Copyright (c) 2025-2026 Jaegeun Han
#
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from datetime import timedelta
from typing import TYPE_CHECKING

import torch
from torch.distributed.fsdp import (
    BackwardPrefetch,
    CPUOffload,
    MixedPrecision,
    ShardingStrategy,
    StateDictType,
)
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    CheckpointImpl,
    apply_activation_checkpointing,
)
from torch.nn.parallel import DistributedDataParallel as DDP

from ironcore import get_logger
from ironcore.config import MainConfig

if TYPE_CHECKING:
    from ironcore.language_model import LanguageModel

# NOTE: FSDP2 (torch.distributed._composable.fsdp) provides additional optimizations
# for CPU offload including pinned memory (CPUOffloadPolicy(pin_memory=True)).
# Current implementation uses FSDP1. FSDP2 support can be added in the future.


def initialize_process(config: MainConfig):

    logger = get_logger()

    # initialize parallelism
    if torch.distributed.is_initialized():
        logger.info(
            f"Torch distributed is already initialized: {torch.distributed.get_world_size()}"
        )
        return

    # initialize cuda
    if torch.cuda.is_available():
        torch.backends.cudnn.enabled = True
        torch.cuda.set_device(config.parallel.local_rank)

        if config.profiler.gpu_profiler:
            torch.backends.cudnn.benchmark = True

        if config.utils.deterministic:
            torch.backends.cudnn.deterministic = True

    # initialize parallelism
    if not torch.cuda.is_available():
        dist_backend = "gloo"
    else:
        dist_backend = config.parallel.dist_backend

    assert torch.distributed.is_available(), "Torch distributed is not available."

    if torch.distributed.is_initialized():
        if config.parallel.rank == 0:
            logger.info(
                f"Torch distributed is already initialized: {torch.distributed.get_world_size()}"
            )
    else:
        if config.parallel.rank == 0:
            logger.info("Initialize torch distributed ... ")

        # initialize torch distributed
        # Set device_id to avoid NCCL warning about guessing device ID
        device_id = None
        if torch.cuda.is_available():
            device_id = torch.device(f"cuda:{torch.cuda.current_device()}")

        torch.distributed.init_process_group(
            backend=dist_backend,
            world_size=config.parallel.world_size,
            rank=config.parallel.rank,
            timeout=timedelta(minutes=config.parallel.timeout_minute),
            device_id=device_id,
        )


def initialize_parallelism(config: MainConfig, model: LanguageModel) -> torch.nn.Module:
    """Initialize DDP or FSDP"""
    from ironcore.parallel import parallel_states

    logger = get_logger()

    if not config.parallel.use_fsdp and config.parallel.world_size >= 1:
        model = DDP(
            model,
            process_group=parallel_states.get_data_parallel_group(),
            broadcast_buffers=False,
        )
    else:
        # FSDP implementation with optimization options
        import functools

        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

        from ironcore.models.transformer import TransformerLayer

        # Layer-wise sharding policy is critical for memory efficiency and AsyncTP overlap
        auto_wrap_policy = functools.partial(
            transformer_auto_wrap_policy,
            transformer_layer_cls={TransformerLayer},
        )

        # mixed precision
        _mixed_precision_opt = MixedPrecision(
            param_dtype=model.dtype,
            reduce_dtype=model.dtype,
            buffer_dtype=model.dtype,
        )
        if config.parallel.fsdp_mixed_precision == "mixed":
            _mixed_precision_opt.reduce_dtype = torch.float32

        # Sharding strategy map including SHARD_GRAD_OP for better CPU offload performance
        _sharding_strategy = {
            "full": ShardingStrategy.FULL_SHARD,
            "hybrid": ShardingStrategy.HYBRID_SHARD,
            "no_shard": ShardingStrategy.NO_SHARD,
            "shard_grad_op": ShardingStrategy.SHARD_GRAD_OP,
        }

        # Prefetching can cause contention with AsyncTP communication
        # Disable forward prefetch if TP > 1 for better stability
        use_forward_prefetch = config.trainer.tensor_model_parallel_size == 1

        # Log FSDP configuration for debugging
        if config.parallel.rank == 0:
            logger.info(
                f"FSDP config: sharding={config.parallel.fsdp_sharding_strategy}, "
                f"offload={config.parallel.fsdp_offload_params}, "
                f"use_orig_params={config.parallel.fsdp_use_orig_params}, "
                f"forward_prefetch={use_forward_prefetch}"
            )

        model = FSDP(
            model,
            process_group=parallel_states.get_data_parallel_group(),
            auto_wrap_policy=auto_wrap_policy,
            cpu_offload=CPUOffload(offload_params=config.parallel.fsdp_offload_params),
            backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
            mixed_precision=_mixed_precision_opt,
            device_id=torch.cuda.current_device(),
            sharding_strategy=_sharding_strategy[config.parallel.fsdp_sharding_strategy],
            forward_prefetch=use_forward_prefetch,
            use_orig_params=config.parallel.fsdp_use_orig_params,
        )

        # Apply FSDP-compatible activation checkpointing
        # Uses module-level checkpointing on TransformerLayer (vs layer-level in transformer.py)
        if config.operation.activation_recompute:
            from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
                checkpoint_wrapper,
            )

            if config.parallel.rank == 0:
                logger.info("Applying FSDP activation checkpointing to TransformerLayer modules")

            # Selective checkpointing: only checkpoint TransformerLayer modules
            def check_fn(submodule):
                return isinstance(submodule, TransformerLayer)

            # Use non-reentrant implementation for better FSDP compatibility
            def wrapper_fn(module):
                return checkpoint_wrapper(module, checkpoint_impl=CheckpointImpl.NO_REENTRANT)

            apply_activation_checkpointing(
                model,
                checkpoint_wrapper_fn=wrapper_fn,
                check_fn=check_fn,
            )

        # Set state dict type separately
        _state_dict_type = {
            "full": StateDictType.FULL_STATE_DICT,
            "local": StateDictType.LOCAL_STATE_DICT,
            "sharded": StateDictType.SHARDED_STATE_DICT,
        }
        from torch.distributed.fsdp import (
            FullStateDictConfig,
            ShardedStateDictConfig,
        )

        st_type = _state_dict_type[config.parallel.fsdp_state_dict_type]
        if st_type == StateDictType.FULL_STATE_DICT:
            FSDP.set_state_dict_type(
                model, st_type, FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
            )
        elif st_type == StateDictType.SHARDED_STATE_DICT:
            FSDP.set_state_dict_type(model, st_type, ShardedStateDictConfig(offload_to_cpu=True))
        else:
            FSDP.set_state_dict_type(model, st_type)

    return model
