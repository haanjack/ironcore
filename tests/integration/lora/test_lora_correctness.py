#!/usr/bin/env python3
# Copyright (c) 2025-2026 Jaegeun Han
#
# SPDX-License-Identifier: Apache-2.0
"""
LoRA TP correctness tests: verify LoRA adapters work correctly with TP=2.

Validates:
1. LoRA parameters are replicated (not sharded) across TP ranks
2. Only LoRA parameters receive gradients (base model frozen)
3. Trainable parameter count is <5% of total with LoRA
4. Forward pass produces valid output under TP=2

Run with:
    torchrun --nproc_per_node=2 -m pytest tests/integration/lora/test_lora_tp_correctness.py -v
"""

import os

import pytest
import torch
import torch.distributed as dist

from tests.fixtures.lora_test_utils import (
    assert_gradient_correctness,
    cleanup_parallel,
    create_lora_test_config,
    create_test_input,
    get_lora_parameters,
    init_parallel,
    print_parameter_stats,
    set_seed,
)

from ironcore.global_vars import global_states_cleanup, set_global_states
from ironcore.language_model import LanguageModel
from ironcore.peft.utils import freeze_base_model

# Skip if not running under torchrun or fewer than 2 GPUs
pytestmark = pytest.mark.skipif(
    "RANK" not in os.environ
    or not torch.cuda.is_available()
    or torch.cuda.device_count() < 2,
    reason="LoRA TP tests require torchrun with at least 2 GPUs",
)


@pytest.fixture(scope="module")
def tp2_lora_config():
    """Initialize TP=2 parallel states and create LoRA config."""
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)

    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")

    init_parallel(tp_size=2)

    config = create_lora_test_config(
        tp_size=2,
        enable_lora=True,
        lora_r=8,
        lora_alpha=16.0,
    )

    set_global_states(config)
    yield config

    global_states_cleanup()
    cleanup_parallel(tp_size=2)


@pytest.fixture(scope="module")
def rank_device():
    """Get current rank and device."""
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)
    return int(os.environ.get("RANK", "0")), device


@pytest.fixture(scope="module")
def lora_model(tp2_lora_config, rank_device):
    """Create and return a LoRA model with frozen base weights."""
    rank, device = rank_device
    set_seed(42)

    model = LanguageModel(tp2_lora_config)
    model.to(device)

    freeze_base_model(model, tp2_lora_config.peft.method)

    yield model

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


class TestLoRATP2Correctness:
    """LoRA adapter correctness tests under TP=2."""

    def test_trainable_ratio_below_five_percent(self, lora_model, tp2_lora_config, rank_device):
        """Verify trainable parameters are <5% of total (LoRA efficiency)."""
        rank, device = rank_device
        trainable, total, lora = print_parameter_stats(lora_model)
        ratio = trainable / total
        assert ratio < 0.05, f"Trainable ratio {100 * ratio:.2f}% exceeds 5%"

    def test_lora_params_have_requires_grad(self, lora_model, tp2_lora_config):
        """Verify LoRA parameters have requires_grad=True and base params do not."""
        for name, param in lora_model.named_parameters():
            if any(k in name for k in ["lora_A", "lora_B"]):
                assert param.requires_grad, f"LoRA param '{name}' should require grad"
            else:
                assert not param.requires_grad, f"Base param '{name}' should not require grad"

    def test_lora_params_replicated_across_ranks(self, lora_model, rank_device):
        """Verify LoRA adapter weights are identical across TP ranks (not sharded)."""
        rank, device = rank_device
        tp_size = dist.get_world_size()

        lora_params = get_lora_parameters(lora_model)

        # Gather LoRA param norms from all ranks
        param_names = list(lora_params.keys())
        assert len(param_names) > 0, "No LoRA parameters found"

        for name in param_names:
            local_param = lora_params[name]
            # LoRA params should be identical across ranks
            gathered = [torch.zeros_like(local_param) for _ in range(tp_size)]
            dist.all_gather(gathered, local_param.contiguous())

            for i in range(1, tp_size):
                diff = torch.abs(gathered[0] - gathered[i]).max().item()
                assert diff == 0.0, (
                    f"LoRA param '{name}' differs between ranks 0 and {i}: max_diff={diff:.2e}"
                )

    def test_forward_pass_valid_output(self, lora_model, tp2_lora_config, rank_device):
        """Verify forward pass produces finite, non-zero output."""
        rank, device = rank_device
        set_seed(42)

        input_ids = create_test_input(2, 16, 100, device, seed=42)
        lora_model.eval()

        with torch.no_grad():
            output = lora_model(input_ids)
        # Handle tuple return
        if isinstance(output, tuple):
            output = output[0]

        assert output is not None, "Model returned None"
        assert torch.isfinite(output).all(), "Output contains non-finite values"
        assert output.abs().sum().item() > 0, "Output is all zeros"
