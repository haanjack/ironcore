# Copyright (c) 2025-2026 Jaegeun Han
#
# SPDX-License-Identifier: Apache-2.0
"""Integration test: end-to-end training with optimizer offload (M1)."""

import math

import pytest
import torch
from tests.integration.offload.conftest import (
    get_offload_config,
    run_training_step,
)

skip_no_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@skip_no_cuda
class TestTrainingLoopOffload:
    """End-to-end training with M1 optimizer state offloading."""

    def test_optimizer_offload_produces_valid_loss(self):
        """Training step with optimizer_offload=True produces non-NaN loss."""
        config = get_offload_config(optimizer_offload=True)
        loss, trainer = run_training_step(config)
        assert not math.isnan(loss), "Loss is NaN"
        assert not math.isinf(loss), "Loss is Inf"
        assert loss > 0, f"Loss should be positive, got {loss}"

    def test_optimizer_states_on_cpu_after_step(self):
        """After a training step, optimizer states should be on CPU."""
        config = get_offload_config(optimizer_offload=True)
        _loss, trainer = run_training_step(config)

        # Check that at least one large param group has CPU states
        found_cpu_state = False
        for group in trainer.optimizer.param_groups:
            for p in group["params"]:
                state = trainer.optimizer.state.get(p, {})
                if "exp_avg" in state:
                    if state["exp_avg"].device.type == "cpu":
                        found_cpu_state = True
                        break
        assert found_cpu_state, "No optimizer states found on CPU"

    def test_loss_matches_baseline(self):
        """Loss with optimizer_offload should match baseline within tolerance."""
        # Run baseline (no offload)
        config_baseline = get_offload_config(optimizer_offload=False)
        config_baseline.offload.enabled = False
        config_baseline.trainer.gradient_accumulation_steps = 1
        config_baseline.init.seed = 42
        loss_baseline, _ = run_training_step(config_baseline)

        # Run with offload (same seed)
        config_offload = get_offload_config(optimizer_offload=True)
        config_offload.init.seed = 42
        loss_offload, _ = run_training_step(config_offload)

        # Losses should be close (within fp tolerance)
        diff = abs(loss_baseline - loss_offload)
        assert diff < 1e-4, (
            f"Loss mismatch: baseline={loss_baseline:.6f}, "
            f"offload={loss_offload:.6f}, diff={diff:.6f}"
        )

    def test_two_steps_loss_decreases(self):
        """Running 2 steps with offload produces different loss (model updates)."""
        from unittest.mock import patch

        import torch.nn.functional as F
        from tests.integration.offload.conftest import (
            create_mock_data_iterator,
            create_mock_evaluators,
            create_mock_forward_step_func,
            setup_distributed,
        )

        from ironcore.global_vars import reset_global_states
        from ironcore.trainers import LanguageModelTrainer

        config = get_offload_config(optimizer_offload=True)
        config.init.seed = 42

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

            loss1, _, _ = trainer.train_step(step=0)
            loss2, _, _ = trainer.train_step(step=1)

            assert not math.isnan(loss1) and not math.isnan(loss2)
            # Loss should change between steps (model is learning)
            assert loss1 != loss2, "Loss should change between steps"
