# Copyright (c) 2025-2026 Jaegeun Han
#
# SPDX-License-Identifier: Apache-2.0

"""
MoE × Offload smoke test.

Verifies MoE (Mixture of Experts) works with offload features.
This is a smoke test - only checks that training runs without errors.
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

NUM_STEPS = 50  # Per test data policy: random data, convergence not expected
BATCH_SIZE = 1  # Smaller batch for MoE memory
SEQ_LEN = 128

# Deterministic seeding
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _make_moe_config(**overrides):
    """Small MoE model config with offload."""
    config = create_test_config(
        d_model=512,  # Smaller model for smoke test
        num_layers=2,  # 2 layers
        num_attention_heads=8,
        num_attention_groups=8,
        head_dim=64,
        max_seq_len=SEQ_LEN,
        dropout_attn=0.0,
        dropout_mlp=0.0,
        dropout_embd=0.0,
        precision="bfloat16",
        seed=42,
    )
    # Enable MoE
    config.model.moe.use_moe = True
    config.model.moe.num_shared_experts = 2
    config.model.moe.num_routed_experts = 8  # Small for smoke test
    config.model.moe.num_experts_per_token = 2
    config.model.moe.expert_intermediate_size = 2048

    config.operation.train_steps = NUM_STEPS + 10
    config.trainer.micro_batch_size = BATCH_SIZE
    config.trainer.train_batch_size = BATCH_SIZE
    config.trainer.gradient_accumulation_steps = 1

    # Disable FSDP for single-GPU tests (avoid NO_SHARD warning)
    config.parallel.use_fsdp = False

    # Apply overrides (offload settings)
    from ironcore.config import OffloadConfig

    offload_overrides = overrides.get("offload", {})
    if not offload_overrides:
        # Default: offload enabled
        offload = OffloadConfig(enabled=True)
    else:
        # Use overrides (which should include enabled=True if desired)
        offload = OffloadConfig(**offload_overrides)
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
    # Set defaults for single-process testing
    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", "29500")
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("LOCAL_RANK", "0")

    config.parallel.rank = 0
    config.parallel.local_rank = 0

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


@skip_no_cuda
class TestMoEOffload:
    """MoE × Offload smoke tests."""

    def test_moe_baseline(self):
        """MoE without offload - baseline for comparison."""
        config = _make_moe_config()

        # Disable offload for baseline
        config.offload.enabled = False

        init_loss, final_loss = _run_training(config, NUM_STEPS)

        print(
            f"\n[MoE+Baseline] Init loss: {init_loss:.4f}, Final loss: {final_loss:.4f}, Reduction: {(init_loss - final_loss) / init_loss * 100:.1f}%"
        )

        assert init_loss is not None
        assert not math.isnan(final_loss) and not math.isinf(final_loss)
        assert final_loss > 0, f"Final loss is invalid: {final_loss:.4f}"

    def test_moe_optimizer_offload(self):
        """MoE + optimizer_offload should run."""
        config = _make_moe_config(
            offload={
                "optimizer_offload": True,
                "optimizer_state_precision": "bf16",
            }
        )

        init_loss, final_loss = _run_training(config, NUM_STEPS)

        print(
            f"\n[MoE+optimizer_offload] Init loss: {init_loss:.4f}, Final loss: {final_loss:.4f}, Reduction: {(init_loss - final_loss) / init_loss * 100:.1f}%"
        )

        assert init_loss is not None
        assert not math.isnan(final_loss) and not math.isinf(final_loss)
        assert final_loss > 0, f"Final loss is invalid: {final_loss:.4f}"

    def test_moe_activation_spill(self):
        """MoE + activation_spill should run."""
        config = _make_moe_config(
            offload={
                "activation_spill": True,
                "activation_spill_granularity": "sub_layer",
            }
        )

        init_loss, final_loss = _run_training(config, NUM_STEPS)

        print(
            f"\n[MoE+activation_spill] Init loss: {init_loss:.4f}, Final loss: {final_loss:.4f}, Reduction: {(init_loss - final_loss) / init_loss * 100:.1f}%"
        )

        assert init_loss is not None
        assert not math.isnan(final_loss) and not math.isinf(final_loss)
        assert final_loss > 0, f"Final loss is invalid: {final_loss:.4f}"

    def test_moe_optimizer_offload_activation_spill(self):
        """MoE + optimizer_offload+activation_spill should run."""
        config = _make_moe_config(
            offload={
                "optimizer_offload": True,
                "activation_spill": True,
                "activation_spill_granularity": "sub_layer",
                "optimizer_state_precision": "bf16",
                "pinned_memory_pool_gb": 2.0,
            }
        )

        init_loss, final_loss = _run_training(config, NUM_STEPS)

        print(
            f"\n[MoE+optimizer_offload+activation_spill] Init loss: {init_loss:.4f}, Final loss: {final_loss:.4f}, Reduction: {(init_loss - final_loss) / init_loss * 100:.1f}%"
        )

        assert init_loss is not None
        assert not math.isnan(final_loss) and not math.isinf(final_loss)
        assert final_loss > 0, f"Final loss is invalid: {final_loss:.4f}"
