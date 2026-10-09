# Copyright (c) 2025-2026 Jaegeun Han
# SPDX-License-Identifier: Apache-2.0

"""Bounded recomputation and fused routing composed with actual CUDA CP2 training."""

from __future__ import annotations

import os

import pytest
import torch
import torch.distributed as dist
from tests.multi_gpu.test_context_parallel import check_model

from ironcore.parallel import parallel_states as ps

pytestmark = [
    pytest.mark.mp,
    pytest.mark.skipif("RANK" not in os.environ, reason="torchrun required"),
]


@pytest.fixture(scope="module")
def parallel_device():
    device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    torch.cuda.set_device(device)
    if not dist.is_initialized():
        dist.init_process_group("nccl", device_id=device)
    # Reuse one mesh: repeated new_group calls retain NCCL communicator storage.
    ps.initialize_model_parallel(1, 2, context_parallel_size=2)
    return device


@pytest.mark.parametrize("expert_backend", ["batched", "grouped"])
@pytest.mark.parametrize("blockwise_backend", ["scheduled", "triton"])
@pytest.mark.parametrize(
    "recompute,lora", [(None, False), ("standard", False), ("optimized", True)]
)
def test_scheduled_context(parallel_device, expert_backend, blockwise_backend, recompute, lora):
    check_model(
        parallel_device,
        "ring",
        lora=lora,
        moe=True,
        mlp_chunk_size=3,
        recompute=recompute,
        compute_dtype=torch.float16,
        expert_backend=expert_backend,
        blockwise_backend=blockwise_backend,
    )
