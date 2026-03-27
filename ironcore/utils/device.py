# Copyright (c) 2025-2026 Jaegeun Han
#
# SPDX-License-Identifier: Apache-2.0

import torch
import torch.distributed as dist


def is_first_rank():
    """Check whether it's rank 0 (or single process)."""
    if not dist.is_initialized():
        return True  # Single process case
    return dist.get_rank() == 0


def is_last_rank():
    """Check whether it's the last rank."""
    assert dist.is_initialized(), "torch distributed is not initialized."
    return dist.get_rank() == dist.get_world_size() - 1


def print_rank_0(message: str):
    """Print message only if it's rank 0."""
    if is_first_rank():
        print(message)


def print_last_rank(message: str):
    """Print message only if it's the last rank."""
    if is_last_rank():
        print(message)


def get_device():
    """Returns device type. Checks MPS first to avoid CUDA context init on Apple Silicon."""
    # Check MPS first (no CUDA side effects)
    if torch.backends.mps.is_available():
        return "mps"

    # Then CUDA
    if torch.cuda.is_available():
        assert torch.distributed.is_initialized(), "torch distributed is not initialized"
        device = (
            f"cuda:{dist.get_node_local_rank()}" if hasattr(dist, "get_node_local_rank") else "cuda"
        )
        return device

    return "cpu"


def get_model_dtype(config):
    """Returns model dtype checking device supports"""
    if config.model.precision.lower() in ["bfloat16", "bf16"]:
        # Check MPS first (no CUDA side effects)
        if torch.backends.mps.is_available():
            dtype = torch.bfloat16  # MPS supports bf16 on Apple Silicon
        elif torch.cuda.is_available():
            assert torch.cuda.is_bf16_supported(), "bfloat16 is not supported on this device"
            dtype = torch.bfloat16
        else:
            dtype = torch.bfloat16  # CPU supports bf16
    elif config.model.precision.lower() in ["float16", "fp16"]:
        dtype = torch.float16
    else:
        # logger.warning("Using FP32, which is slow for the training.")
        dtype = torch.float

    return dtype
