# Copyright (c) 2025-2026 Jaegeun Han
#
# SPDX-License-Identifier: Apache-2.0
"""Shared fixtures for offload integration tests."""

import os
from unittest.mock import patch

import pytest
import torch
import torch.distributed as dist
import torch.nn.functional as F
from tests.fixtures.config_fixtures import create_small_test_config

from ironcore.global_vars import reset_global_states


def setup_distributed():
    """Initialize single-GPU distributed environment."""
    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", "29500")
    os.environ.setdefault("LOCAL_RANK", "0")
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl", rank=0, world_size=1)


def create_mock_forward_step_func(batch_size=2, seq_len=16):
    """Create a forward step function that generates random input and computes loss."""

    def mock_forward_step(model, data_iterator):
        device = next(model.parameters()).device
        input_ids = torch.randint(0, 1000, (batch_size, seq_len), device=device)
        labels = input_ids.clone()
        logits = model(input_ids, labels=None)
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()
        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
        )
        return loss

    return mock_forward_step


def create_mock_data_iterator():
    """Create a mock data iterator that yields nothing."""
    return {
        "train": iter([]),
        "eval": iter([]),
        "test": iter([]),
    }


def create_mock_evaluators():
    """Return empty evaluators list."""
    return []


def get_offload_config(**overrides):
    """Create a small test config with offload enabled."""
    config = create_small_test_config()
    config.offload.enabled = True
    config.trainer.gradient_accumulation_steps = 1
    for k, v in overrides.items():
        setattr(config.offload, k, v)
    return config


def run_training_step(config):
    """Run a single training step and return (loss, model, optimizer)."""
    from ironcore.trainers import LanguageModelTrainer

    reset_global_states()
    setup_distributed()

    with (
        patch(
            "ironcore.trainers.base_trainer.get_data_iterator",
            return_value=create_mock_data_iterator(),
        ),
        patch(
            "ironcore.trainers.base_trainer.get_evaluators",
            return_value=create_mock_evaluators(),
        ),
    ):
        trainer = LanguageModelTrainer(
            config,
            create_mock_forward_step_func(),
            F.cross_entropy,
        )
        trainer._initialize()
        loss, grad_norm, param_norm = trainer.train_step(step=0)
        return loss, trainer


@pytest.fixture(autouse=True)
def reset_state():
    """Reset global state before and after each test."""
    reset_global_states()
    yield
    reset_global_states()
