# Copyright (c) 2025-2026 Jaegeun Han
#
# SPDX-License-Identifier: Apache-2.0
"""
Combinational training loss validation for offload modes.

Runs 100 training steps through the real LanguageModelTrainer pipeline
(LanguageModel with embeddings + output head, cross-entropy loss,
GradScaler, autocast) for each offload combination and compares final
losses against a no-offload baseline.

Model: GPT-small architecture (4L, 768d, 3072 FFN, 12 heads).

Combinations tested:
  - M1 only (optimizer offload)
  - M2 + M3 (weight streaming + activation spill)
  - M1 + M2 + M3 (all three)
  - M2 + M3 with grad_accum=2
  - M1 + M2 + M3 with full_layer granularity
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


def _make_config(**offload_overrides):
    """GPT-small architecture config with optional offload settings."""
    from ironcore.offload.config import OffloadConfig

    offload = OffloadConfig()
    for k, v in offload_overrides.items():
        setattr(offload, k, v)

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
    """Deterministic forward step: step-varying seed for diverse data each step."""

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
    """Run N training steps through the real pipeline. Returns (initial_loss, final_loss)."""
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
            _create_forward_step_func(),
            F.cross_entropy,
        )
        trainer._initialize()

        for step in range(num_steps):
            loss, _, _ = trainer.train_step(step=step)
            if step == 0:
                initial_loss = loss
            final_loss = loss

        trainer._finalize_process()

    return initial_loss, final_loss


@skip_no_cuda
class TestCombinationalTraining:
    """100-step training loss parity across offload mode combinations."""

    def test_m1_only(self):
        """M1 (optimizer offload): final loss must match baseline."""
        _, loss_ref = _run_training(_make_config(), NUM_STEPS)
        _, loss_off = _run_training(_make_config(optimizer_offload=True), NUM_STEPS)

        assert not math.isnan(loss_off) and not math.isinf(loss_off)
        rel_err = abs(loss_ref - loss_off) / (abs(loss_ref) + 1e-8)
        assert rel_err < 0.01, (
            f"M1 final loss diverged after {NUM_STEPS} steps: "
            f"ref={loss_ref:.4f} off={loss_off:.4f} rel_err={rel_err:.4f}"
        )

    def test_m2_m3(self):
        """M2 + M3: final loss must be within tolerance of baseline."""
        _, loss_ref = _run_training(_make_config(), NUM_STEPS)
        _, loss_off = _run_training(
            _make_config(
                weight_offload=True,
                activation_spill=True,
                activation_spill_granularity="sub_layer",
                pinned_chunk_gb=0.1,
                pinned_memory_pool_gb=1.0,
                gpu_staging_chunk_mb=64.0,
                gpu_staging_pool_mb=0,
            ),
            NUM_STEPS,
        )

        assert not math.isnan(loss_off) and not math.isinf(loss_off)
        rel_err = abs(loss_ref - loss_off) / (abs(loss_ref) + 1e-8)
        assert rel_err < 0.05, (
            f"M2+M3 final loss diverged after {NUM_STEPS} steps: "
            f"ref={loss_ref:.4f} off={loss_off:.4f} rel_err={rel_err:.4f}"
        )

    def test_m1_m2_m3(self):
        """M1 + M2 + M3 (all offload): final loss must stay close to baseline."""
        _, loss_ref = _run_training(_make_config(), NUM_STEPS)
        _, loss_off = _run_training(
            _make_config(
                optimizer_offload=True,
                weight_offload=True,
                activation_spill=True,
                activation_spill_granularity="sub_layer",
                weight_storage_precision="bf16",
                optimizer_state_precision="bf16",
                pinned_chunk_gb=0.1,
                pinned_memory_pool_gb=1.0,
                gpu_staging_chunk_mb=64.0,
                gpu_staging_pool_mb=0,
            ),
            NUM_STEPS,
        )

        assert not math.isnan(loss_off) and not math.isinf(loss_off)
        rel_err = abs(loss_ref - loss_off) / (abs(loss_ref) + 1e-8)
        assert rel_err < 0.05, (
            f"M1+M2+M3 final loss diverged after {NUM_STEPS} steps: "
            f"ref={loss_ref:.4f} off={loss_off:.4f} rel_err={rel_err:.4f}"
        )

    def test_m2_m3_grad_accum(self):
        """M2 + M3 with grad_accum=2: loss must track baseline."""
        config_ref = _make_config()
        config_ref.trainer.gradient_accumulation_steps = 2
        config_ref.trainer.train_batch_size = BATCH_SIZE * 2

        config_off = _make_config(
            weight_offload=True,
            activation_spill=True,
            activation_spill_granularity="sub_layer",
            pinned_chunk_gb=0.1,
            pinned_memory_pool_gb=1.0,
            gpu_staging_chunk_mb=64.0,
            gpu_staging_pool_mb=0,
        )
        config_off.trainer.gradient_accumulation_steps = 2
        config_off.trainer.train_batch_size = BATCH_SIZE * 2

        init_ref, final_ref = _run_training(config_ref, NUM_STEPS)
        init_off, final_off = _run_training(config_off, NUM_STEPS)

        assert not math.isnan(final_off) and not math.isinf(final_off)
        assert init_off is not None
        assert final_off < init_off, (
            f"M2+M3+grad_accum did not converge: {init_off:.4f} -> {final_off:.4f}"
        )
        rel_err = abs(final_ref - final_off) / (abs(final_ref) + 1e-8)
        assert rel_err < 0.05, (
            f"M2+M3+grad_accum final loss diverged: "
            f"ref={final_ref:.4f} off={final_off:.4f} rel_err={rel_err:.4f}"
        )

    def test_m1_m2_m3_full_layer(self):
        """M1 + M2 + M3 with full_layer granularity: final loss must track baseline."""
        _, loss_ref = _run_training(_make_config(), NUM_STEPS)
        _, loss_off = _run_training(
            _make_config(
                optimizer_offload=True,
                weight_offload=True,
                activation_spill=True,
                activation_spill_granularity="full_layer",
                weight_storage_precision="bf16",
                optimizer_state_precision="bf16",
                pinned_chunk_gb=0.1,
                pinned_memory_pool_gb=1.0,
                gpu_staging_chunk_mb=64.0,
                gpu_staging_pool_mb=0,
            ),
            NUM_STEPS,
        )

        assert not math.isnan(loss_off) and not math.isinf(loss_off)
        rel_err = abs(loss_ref - loss_off) / (abs(loss_ref) + 1e-8)
        assert rel_err < 0.05, (
            f"full_layer final loss diverged after {NUM_STEPS} steps: "
            f"ref={loss_ref:.4f} off={loss_off:.4f} rel_err={rel_err:.4f}"
        )
