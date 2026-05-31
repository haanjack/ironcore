# Copyright (c) 2025-2026 Jaegeun Han
#
# SPDX-License-Identifier: Apache-2.0
"""Integration test: weight streaming with DP=2 (ZeRO-3 parameter sharding).

Requires 2 GPUs via torchrun:
    torchrun --nproc_per_node 2 -m pytest tests/integration/offload/test_weight_streaming_dp.py -v
"""

import math
import os

import pytest
import torch
import torch.distributed as dist

skip_no_cuda = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.device_count() < 2,
    reason="2 CUDA GPUs required",
)

pytestmark = [pytest.mark.cuda, pytest.mark.mp]


def _setup_dp2():
    """Initialize DP=2 distributed environment."""
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "2"))

    torch.cuda.set_device(local_rank)
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)


def _make_dp2_offload_config():
    """Create a small config for DP=2 + weight streaming (ZeRO-3)."""
    from tests.fixtures.config_fixtures import create_small_test_config

    config = create_small_test_config()
    config.offload.enabled = True
    config.offload.weight_offload = True
    config.trainer.tensor_model_parallel_size = 1
    config.parallel.world_size = 2
    config.trainer.micro_batch_size = 1
    config.trainer.train_batch_size = 2  # 2 DP ranks * 1 micro_batch
    config.trainer.gradient_accumulation_steps = 1
    config.init.seed = 42
    return config


@skip_no_cuda
class TestWeightStreamingDP2:
    """Weight streaming with ZeRO-3 parameter sharding (DP=2, TP=1)."""

    def test_dp2_forward_valid_output(self):
        """Forward pass with DP=2 + ZeRO-3 produces valid output."""
        from ironcore.global_vars import initialize_global_states, reset_global_states
        from ironcore.parallel import initialize_process, parallel_states

        reset_global_states()
        _setup_dp2()

        config = _make_dp2_offload_config()
        initialize_global_states(config)
        initialize_process(config)
        if not parallel_states.is_model_parallel_initialized():
            parallel_states.initialize_model_parallel(config.trainer.tensor_model_parallel_size)

        from ironcore.language_model import LanguageModel
        from ironcore.utils.device import get_device

        model = LanguageModel(config, torch.nn.functional.cross_entropy)
        model = model.to(dtype=torch.bfloat16)
        gpu = torch.device(get_device())

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

    def test_dp2_training_step_no_leaks(self):
        """Training step with DP=2 + ZeRO-3 runs without memory leaks."""
        from unittest.mock import patch

        from ironcore.global_vars import initialize_global_states, reset_global_states
        from ironcore.parallel import initialize_process, parallel_states
        from ironcore.trainers import LanguageModelTrainer

        reset_global_states()
        _setup_dp2()

        config = _make_dp2_offload_config()
        initialize_global_states(config)
        initialize_process(config)
        if not parallel_states.is_model_parallel_initialized():
            parallel_states.initialize_model_parallel(config.trainer.tensor_model_parallel_size)

        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        with (
            patch(
                "ironcore.trainers.base_trainer.get_data_iterator",
                return_value={"train": iter([]), "eval": iter([]), "test": iter([])},
            ),
            patch("ironcore.trainers.base_trainer.get_evaluators", return_value=[]),
        ):

            def forward_step(model, _data_iterator):
                device = next(model.parameters()).device
                input_ids = torch.randint(0, 100, (1, 8), device=device)
                labels = input_ids.clone()
                return model(input_ids, labels=labels)

            trainer = LanguageModelTrainer(config, forward_step, torch.nn.functional.cross_entropy)
            trainer._initialize()

            peak_before = torch.cuda.max_memory_allocated() / 1e9
            loss, _grad_norm, _param_norm = trainer.train_step(step=0)
            peak_after = torch.cuda.max_memory_allocated() / 1e9

        assert not math.isnan(loss), "Loss is NaN"
        assert loss > 0, f"Loss should be positive, got {loss}"
        # Peak memory should not grow significantly during step
        assert peak_after < peak_before * 2.0, (
            f"Memory leak suspected: peak grew from {peak_before:.2f}GB to {peak_after:.2f}GB"
        )

        parallel_states.destroy_model_parallel()
        reset_global_states()
        if dist.is_initialized():
            dist.destroy_process_group()
