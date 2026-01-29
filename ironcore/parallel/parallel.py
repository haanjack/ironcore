from __future__ import annotations

from datetime import timedelta
from typing import TYPE_CHECKING

import torch
from torch import distributed as dist
from torch.distributed.fsdp import (
    BackwardPrefetch,
    CPUOffload,
    MixedPrecision,
    ShardingStrategy,
    StateDictType,
)
from torch.distributed.fsdp.wrap import wrap
from torch.nn.parallel import DistributedDataParallel as DDP

from ironcore import get_logger
from ironcore.config import MainConfig

if TYPE_CHECKING:
    from ironcore.language_model import LanguageModel

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

        if config.utils.profile_nsys:
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
        torch.distributed.init_process_group(
            backend=dist_backend,
            world_size=config.parallel.world_size,
            rank=config.parallel.rank,
            timeout=timedelta(minutes=config.parallel.timeout_minute),
        )


def initialize_parallelism(config: MainConfig, model: "LanguageModel") -> torch.nn.Module:
    """Initialize DDP or FSDP"""
    from ironcore.parallel import parallel_states

    if not config.parallel.use_fsdp and config.parallel.world_size >= 1:
        model = DDP(
            model,
            process_group=parallel_states.get_data_parallel_group(),
            broadcast_buffers=False,
        )
    else:
        # initailize FSDP configs
        _mixed_precision_opt = MixedPrecision(
            param_dtype=model.dtype,
            reduce_dtype=model.dtype,
            buffer_dtype=model.dtype,
        )
        if config.parallel.fsdp_mixed_precision == "mixed":
            _mixed_precision_opt.reduce_dtype = float

        _state_dict_type = {
            "full": StateDictType.FULL_STATE_DICT,
            "local": StateDictType.LOCAL_STATE_DICT,
            "sharded": StateDictType.SHARDED_STATE_DICT,
        }
        _sharding_strategy = {
            "full": ShardingStrategy.FULL_SHARD,
            "hybrid": ShardingStrategy.HYBRID_SHARD,
            "no_shard": ShardingStrategy.NO_SHARD,
        }
        fsdp_config = {
            "cpu_offload": CPUOffload(
                offload_params=config.parallel.fsdp_offload_params
            ),
            "forward_prefetch": True,
            "backward_prefetch": BackwardPrefetch.BACKWARD_PRE,
            "mixed_precision": _mixed_precision_opt,
            "device_id": torch.cuda.current_device(),
            "sharding_strategy": _sharding_strategy[
                config.parallel.fsdp_sharding_strategy
            ],
            "state_dict_type": _state_dict_type[config.parallel.fsdp_state_dict_type],
        }
        model = wrap(model, **fsdp_config)

    return model
