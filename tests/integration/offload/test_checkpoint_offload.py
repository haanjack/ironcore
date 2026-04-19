# Copyright (c) 2025-2026 Jaegeun Han
#
# SPDX-License-Identifier: Apache-2.0
"""Integration test: checkpoint save/load with offloaded optimizer states."""

import math
import os
import tempfile
from unittest.mock import patch

import pytest
import torch
import torch.nn.functional as F

from ironcore.global_vars import reset_global_states
from ironcore.trainers import LanguageModelTrainer
from tests.fixtures.config_fixtures import create_small_test_config
from tests.integration.offload.conftest import (
    create_mock_data_iterator,
    create_mock_evaluators,
    create_mock_forward_step_func,
    get_offload_config,
    run_training_step,
    setup_distributed,
)

skip_no_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA required"
)


@skip_no_cuda
class TestCheckpointOffload:
    """Checkpoint save/load with offloaded optimizer states."""

    def test_save_load_roundtrip(self):
        """Save and load checkpoint with offloaded optimizer states."""
        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_path = os.path.join(tmpdir, "offload_test_ckpt")

            # Step 1: Train with offload, save checkpoint
            config = get_offload_config(optimizer_offload=True)
            config.init.seed = 42
            config.trainer.model_path = ckpt_path
            config.operation.train_steps = 2
            config.trainer.gradient_accumulation_steps = 1
            config.model.hf_model_type = "gpt2"
            config.model.hf_architecture = "GPT2LMHeadModel"

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
                trainer1 = LanguageModelTrainer(
                    config,
                    create_mock_forward_step_func(),
                    F.cross_entropy,
                )
                trainer1._initialize()

                loss1_step0, _, _ = trainer1.train_step(step=0)
                loss1_step1, _, _ = trainer1.train_step(step=1)

                # Save checkpoint
                from ironcore.checkpointing import save_checkpoint
                save_checkpoint(config, trainer1.model, trainer1.optimizer, trainer1.lr_scheduler, 1)

            # Step 2: Load checkpoint and continue training
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
                trainer2 = LanguageModelTrainer(
                    config,
                    create_mock_forward_step_func(),
                    F.cross_entropy,
                )
                trainer2._initialize()

                # Load checkpoint
                from ironcore.checkpointing import load_checkpoint
                loaded_step = load_checkpoint(config, trainer2.model, trainer2.optimizer, trainer2.lr_scheduler)
                assert loaded_step == 1, f"Expected step 1, got {loaded_step}"

                # Continue training
                loss2_step1, _, _ = trainer2.train_step(step=1)

                assert not math.isnan(loss2_step1), "Loss after load is NaN"
                assert loss2_step1 > 0, f"Loss should be positive, got {loss2_step1}"

    def test_save_succeeds_with_cpu_states(self):
        """Checkpoint save succeeds even when optimizer states are on CPU."""
        config = get_offload_config(optimizer_offload=True)
        config.model.hf_model_type = "gpt2"
        config.model.hf_architecture = "GPT2LMHeadModel"

        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_path = os.path.join(tmpdir, "cpu_state_ckpt")
            config.trainer.model_path = ckpt_path

            _loss, trainer = run_training_step(config)

            from ironcore.checkpointing import save_checkpoint
            # Should not raise
            save_checkpoint(config, trainer.model, trainer.optimizer, trainer.lr_scheduler, 1)

            # Verify checkpoint files exist
            assert os.path.exists(ckpt_path) or os.path.exists(ckpt_path + ".pt"), \
                f"Checkpoint not found at {ckpt_path}"
