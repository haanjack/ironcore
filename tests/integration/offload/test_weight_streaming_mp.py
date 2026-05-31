# Copyright (c) 2025-2026 Jaegeun Han
#
# SPDX-License-Identifier: Apache-2.0
"""Integration test: weight streaming with TP=2.

Requires 2 GPUs via torchrun:
    torchrun --nproc_per_node 2 -m pytest tests/integration/offload/test_weight_streaming_tp2.py -v
"""

import math
import os

import pytest
import torch
import torch.distributed as dist

from ironcore.global_vars import reset_global_states

skip_no_cuda = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.device_count() < 2,
    reason="2 CUDA GPUs required",
)

pytestmark = [pytest.mark.cuda, pytest.mark.mp]


def _setup_tp2():
    """Initialize TP=2 distributed environment."""
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "2"))

    torch.cuda.set_device(local_rank)
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)


def _make_tp2_offload_config():
    """Create a small config for TP=2 + weight streaming."""
    from tests.fixtures.config_fixtures import create_small_test_config

    config = create_small_test_config()
    config.offload.enabled = True
    config.offload.weight_offload = True
    config.trainer.tensor_model_parallel_size = 2
    config.parallel.world_size = 2
    config.trainer.micro_batch_size = 1
    config.trainer.train_batch_size = 1
    config.trainer.gradient_accumulation_steps = 1
    config.init.seed = 42
    return config


@skip_no_cuda
class TestWeightStreamingTP2:
    """Weight streaming with tensor parallelism (TP=2, DP=1)."""

    def test_tp2_weight_streaming_forward_pass(self):
        """Forward pass with TP=2 + weight streaming produces valid output."""
        from ironcore.global_vars import initialize_global_states
        from ironcore.parallel import initialize_process, parallel_states

        reset_global_states()
        _setup_tp2()

        config = _make_tp2_offload_config()
        initialize_global_states(config)
        initialize_process(config)
        if not parallel_states.is_model_parallel_initialized():
            parallel_states.initialize_model_parallel(config.trainer.tensor_model_parallel_size)

        from ironcore.language_model import LanguageModel
        from ironcore.utils.device import get_device

        # Create model on CPU, then move embedding to GPU
        model = LanguageModel(config, torch.nn.functional.cross_entropy)
        model = model.to(dtype=torch.bfloat16)
        gpu = torch.device(get_device())
        model.embedding = model.embedding.to(gpu)
        model.output_layernorm = model.output_layernorm.to(gpu)

        # Verify transformer layers are on CPU
        first_layer = model.model.layers[0]
        param_device = next(first_layer.parameters()).device
        assert param_device.type == "cpu", f"Layer params should be on CPU, got {param_device}"

        # Verify embedding is on GPU
        emb_device = next(model.embedding.parameters()).device
        assert emb_device.type == "cuda", f"Embedding should be on CUDA, got {emb_device}"

        # Forward pass
        batch_size, seq_len = 1, 8
        input_ids = torch.randint(0, 100, (batch_size, seq_len), device=gpu)
        labels = input_ids.clone()

        with torch.no_grad():
            loss = model(input_ids, labels=labels)

        assert not math.isnan(loss), "Loss is NaN"
        assert loss > 0, f"Loss should be positive, got {loss}"

        parallel_states.destroy_model_parallel()
        reset_global_states()
        if dist.is_initialized():
            dist.destroy_process_group()
