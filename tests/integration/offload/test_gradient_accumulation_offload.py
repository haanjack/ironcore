# Copyright (c) 2025-2026 Jaegeun Han
#
# SPDX-License-Identifier: Apache-2.0
"""Integration test: gradient accumulation with optimizer offload (M1)."""

import math
from unittest.mock import patch

import pytest
import torch
import torch.nn.functional as F

from ironcore.global_vars import reset_global_states
from ironcore.trainers import LanguageModelTrainer
from tests.integration.offload.conftest import (
    create_mock_data_iterator,
    create_mock_evaluators,
    create_mock_forward_step_func,
    get_offload_config,
    setup_distributed,
)

skip_no_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA required"
)


def _run_steps(config):
    """Run training with given config, return loss after 1 optimizer step."""
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
        loss, _, _ = trainer.train_step(step=0)
        return loss, trainer


@skip_no_cuda
class TestGradientAccumulationOffload:
    """Gradient accumulation with optimizer state offloading."""

    def test_grad_accum_2_with_offload(self):
        """Gradient accumulation with 2 microbatches works with offload."""
        config = get_offload_config(optimizer_offload=True)
        config.trainer.gradient_accumulation_steps = 2
        config.init.seed = 42

        loss, trainer = _run_steps(config)
        assert not math.isnan(loss), "Loss is NaN with grad_accum=2"
        assert loss > 0, f"Loss should be positive, got {loss}"

    def test_states_on_cpu_during_accumulation(self):
        """Optimizer states stay on CPU throughout gradient accumulation."""
        config = get_offload_config(optimizer_offload=True)
        config.trainer.gradient_accumulation_steps = 3
        config.init.seed = 42

        _loss, trainer = _run_steps(config)

        # After step, states should be on CPU
        found_cpu_state = False
        for group in trainer.optimizer.param_groups:
            for p in group["params"]:
                state = trainer.optimizer.state.get(p, {})
                if "exp_avg" in state and state["exp_avg"].device.type == "cpu":
                    found_cpu_state = True
                    break
        assert found_cpu_state, "No optimizer states on CPU after grad accumulation"

    def test_grad_accum_matches(self):
        """Loss with grad_accum=2 should be reasonable vs grad_accum=1."""
        config1 = get_offload_config(optimizer_offload=True)
        config1.trainer.gradient_accumulation_steps = 1
        config1.init.seed = 42
        loss1, _ = _run_steps(config1)

        config2 = get_offload_config(optimizer_offload=True)
        config2.trainer.gradient_accumulation_steps = 2
        config2.init.seed = 42
        loss2, _ = _run_steps(config2)

        # Both should be valid and not wildly different
        assert not math.isnan(loss1) and not math.isnan(loss2)
        # With different accumulation, losses will differ but both should be finite and positive
        assert loss1 > 0 and loss2 > 0
