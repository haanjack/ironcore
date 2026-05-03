# Copyright (c) 2025-2026 Jaegeun Han
#
# SPDX-License-Identifier: Apache-2.0

"""
Pairwise offload test: M1 + M2 (M3 auto-enabled by M2).

Verifies M1+M2+M3 combination works correctly. Note: M2 automatically
enables M3 for weight eviction safety, so this tests the full stack.
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
skip_no_cuda = pytest.mark.skipif(not cuda_available, reason="CUDA not available")

NUM_STEPS = 50
BATCH_SIZE = 2
SEQ_LEN = 256

# Deterministic seeding
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _make_config(**offload_overrides):
    """GPT-small architecture config with M1+M2 offload."""
    from ironcore.offload.config import OffloadConfig

    offload = OffloadConfig(enabled=True, **offload_overrides)
    config = create_test_config(
        d_model=768,
        d_ffn=3072,
        num_layers=4,
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
    config.trainer.micro_batch_size = BATCH_SIZE
    config.trainer.train_batch_size = BATCH_SIZE
    config.trainer.gradient_accumulation_steps = 1
    config.parallel.world_size = 1
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
        logits = model(input_ids, labels=None)
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
    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", "29500")
    os.environ.setdefault("LOCAL_RANK", "0")
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    if not torch.distributed.is_initialized():
        torch.distributed.init_process_group(backend="nccl", rank=0, world_size=1)

    initial_loss: float | None = None
    final_loss = 0.0

    with (
        patch("ironcore.trainers.base_trainer.get_data_iterator", return_value=create_mock_data_iterator()),
        patch("ironcore.trainers.base_trainer.get_evaluators", return_value=create_mock_evaluators()),
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


@skip_no_cuda
class TestPairwiseM1M2:
    """M1 + M2 (M3 auto-enabled) combination test."""

    def test_m1_m2_converges(self):
        """M1+M2+M3 should converge and produce valid gradients."""
        config = _make_config(
            optimizer_offload=True,
            weight_offload=True,
            weight_storage_precision="bf16",
            optimizer_state_precision="bf16",
            pinned_memory_pool_gb=2.0,
            gpu_staging_pool_mb=0,
        )

        init_loss, final_loss = _run_training(config, NUM_STEPS)

        assert init_loss is not None
        assert not math.isnan(final_loss) and not math.isinf(final_loss)
        assert final_loss < init_loss, f"M1+M2 did not converge: {init_loss:.4f} -> {final_loss:.4f}"
        assert final_loss > 0, f"Final loss is invalid: {final_loss:.4f}"

    def test_m1_m2_vs_baseline_parity(self):
        """M1+M2+M3 final loss should match baseline within tolerance."""
        _, loss_ref = _run_training(_make_config(), NUM_STEPS)
        _, loss_off = _run_training(
            _make_config(
                optimizer_offload=True,
                weight_offload=True,
                weight_storage_precision="bf16",
                optimizer_state_precision="bf16",
                pinned_memory_pool_gb=2.0,
                gpu_staging_pool_mb=0,
            ),
            NUM_STEPS,
        )

        rel_err = abs(loss_ref - loss_off) / (abs(loss_ref) + 1e-8)
        assert rel_err < 0.01, (
            f"M1+M2 loss diverged: ref={loss_ref:.4f} off={loss_off:.4f} rel_err={rel_err:.4f}"
        )
