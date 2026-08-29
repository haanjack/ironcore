# Copyright (c) 2025-2026 Jaegeun Han
#
# SPDX-License-Identifier: Apache-2.0

"""
DDP × Offload integration test.

Verifies DDP + optimizer_offload+activation_spill works correctly with 2 GPUs via torchrun.
Each rank independently offloads its optimizer states to host RAM.
"""

import math
import os
from unittest.mock import patch

import pytest
import torch
import torch.nn.functional as F
from tests.fixtures.config_fixtures import create_test_config
from tests.integration.offload.conftest import (
    create_mock_data_iterator,
    create_mock_evaluators,
)

from ironcore.global_vars import reset_global_states
from ironcore.trainers import LanguageModelTrainer

cuda_available = torch.cuda.is_available()
has_multi_gpu = (
    cuda_available and torch.cuda.device_count() >= 2 and os.environ.get("RANK") is not None
)
skip_no_multi_gpu = pytest.mark.skipif(
    not has_multi_gpu,
    reason="Requires torchrun with 2+ GPUs",
)

NUM_STEPS = 50
BATCH_SIZE = 2
SEQ_LEN = 256

# Deterministic seeding
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _make_config(**overrides):
    """GPT-small architecture config with DDP+offload."""
    config = create_test_config(
        d_model=768,
        d_ffn=3072,
        num_layers=4,  # Same as single-GPU tests
        num_attention_heads=12,
        num_attention_groups=12,
        head_dim=64,
        max_seq_len=SEQ_LEN,
        dropout_attn=0.0,
        dropout_mlp=0.0,
        dropout_embd=0.0,
        precision="bfloat16",
        seed=42,
    )
    config.operation.train_steps = NUM_STEPS + 10
    config.trainer.micro_batch_size = 1  # Per-rank batch
    config.trainer.train_batch_size = 2  # Global batch = 2 (same as single-GPU)
    config.trainer.gradient_accumulation_steps = 1
    config.parallel.world_size = 2

    # Apply overrides (offload settings)
    from ironcore.config import OffloadConfig

    offload = OffloadConfig(enabled=True, **overrides.get("offload", {}))
    config.offload = offload

    return config


def _create_forward_step_func():
    """Deterministic forward step."""

    step_counter = [0]

    def forward_step(model, _data_iterator):
        device = next(model.parameters()).device
        torch.manual_seed(42 + step_counter[0])
        step_counter[0] += 1
        input_ids = torch.randint(0, 1000, (BATCH_SIZE, SEQ_LEN), device=device)
        labels = input_ids.clone()
        logits, _ = model(input_ids, labels=None)
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()
        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
        )
        return loss

    return forward_step


def _run_training(config, num_steps):
    """Run N training steps. Returns (initial_loss, final_loss)."""
    reset_global_states()
    # torchrun sets these; only set defaults for single-process testing
    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", "29500")
    os.environ.setdefault("WORLD_SIZE", "2")

    # Ensure config picks up the actual rank from environment
    config.parallel.rank = int(os.getenv("RANK", "0"))
    config.parallel.local_rank = int(os.getenv("LOCAL_RANK", "0"))

    initial_loss: float | None = None
    final_loss = 0.0

    with (
        patch(
            "ironcore.trainers.base_trainer.get_data_iterator",
            return_value=create_mock_data_iterator(),
        ),
        patch(
            "ironcore.trainers.base_trainer.get_evaluators", return_value=create_mock_evaluators()
        ),
    ):
        trainer = LanguageModelTrainer(config, _create_forward_step_func(), F.cross_entropy)
        trainer._initialize()

        for step in range(num_steps):
            loss, _, _ = trainer.train_step(step=step)
            if step == 0:
                initial_loss = loss
            final_loss = loss

        trainer._finalize_process()

    return initial_loss, final_loss


@skip_no_multi_gpu
@pytest.mark.cuda
@pytest.mark.mp
class TestDDPOffload:
    """DDP × Offload integration test (requires 2 GPUs)."""

    def test_ddp_optimizer_with_spill(self):
        """DDP + optimizer_offload+activation_spill should converge on both ranks."""
        config = _make_config(
            offload={
                "optimizer_offload": True,
                "activation_spill": True,
                "activation_spill_granularity": "sub_layer",
                "optimizer_state_precision": "bf16",
                "pinned_memory_pool_gb": 2.0,
            }
        )

        init_loss, final_loss = _run_training(config, NUM_STEPS)

        rank = int(os.getenv("RANK", "0"))
        if rank == 0:
            print(
                f"\n[DDP+optimizer_offload+activation_spill] Init loss: {init_loss:.4f}, Final loss: {final_loss:.4f}, Reduction: {(init_loss - final_loss) / init_loss * 100:.1f}%"
            )

        assert init_loss is not None
        assert not math.isnan(final_loss) and not math.isinf(final_loss)
        assert final_loss < init_loss, (
            f"DDP+optimizer_offload+activation_spill did not converge: {init_loss:.4f} -> {final_loss:.4f}"
        )
        assert final_loss > 0, f"Final loss is invalid: {final_loss:.4f}"
