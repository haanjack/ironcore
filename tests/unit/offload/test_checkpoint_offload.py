# Copyright (c) 2025-2026 Jaegeun Han
#
# SPDX-License-Identifier: Apache-2.0
"""Tests for checkpoint save with missing HF config fields (BUG-005 regression)."""

import os
import tempfile
from unittest.mock import MagicMock, patch

import pytest
import torch
from tests.fixtures.config_fixtures import create_small_test_config
from torch import nn

cuda_available = torch.cuda.is_available()
skip_no_cuda = pytest.mark.skipif(not cuda_available, reason="CUDA not available")

pytestmark = [pytest.mark.cuda]


def _make_config(no_save=True, model_path=""):
    config = create_small_test_config()
    config.offload.enabled = False
    config.operation.no_save = no_save
    config.trainer.model_path = model_path
    return config


class _SimpleModel(nn.Module):
    """Minimal nn.Module that satisfies save_checkpoint's duck-typed interface."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.linear = nn.Linear(config.model.d_model, config.model.d_model)


@skip_no_cuda
class TestCheckpointSaveHFConfig:
    """BUG-005: save_checkpoint should not crash when hf_model_type is None."""

    def test_save_no_hf_fields(self):
        """save_checkpoint should succeed without hf_model_type/hf_architecture."""
        from ironcore.checkpointing.native import save_checkpoint

        with tempfile.TemporaryDirectory() as tmpdir:
            config = _make_config(no_save=False, model_path=tmpdir)
            config.model.hf_model_type = None
            config.model.hf_architecture = None

            model = _SimpleModel(config)
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10)

            import logging

            mock_logger = logging.getLogger("test_ckpt")
            mock_timer = MagicMock()
            mock_timer.get.return_value = 0.0
            with (
                patch("ironcore.checkpointing.native.get_logger", return_value=mock_logger),
                patch("ironcore.checkpointing.native.get_timer", return_value=mock_timer),
            ):
                # Should NOT raise ValueError
                save_checkpoint(config, model, optimizer, scheduler, step=1)

            ckpt_file = os.path.join(tmpdir, "step_1", "pytorch_model.bin")
            assert os.path.exists(ckpt_file), f"Checkpoint not found at {ckpt_file}"

    def test_save_with_hf_fields(self):
        """save_checkpoint should still work and save config.json when HF fields are set."""
        from ironcore.checkpointing.native import save_checkpoint

        with tempfile.TemporaryDirectory() as tmpdir:
            config = _make_config(no_save=False, model_path=tmpdir)
            config.model.hf_model_type = "gpt2"
            config.model.hf_architecture = "GPT2LMHeadModel"

            model = _SimpleModel(config)
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10)

            import logging

            mock_logger = logging.getLogger("test_ckpt")
            mock_timer = MagicMock()
            mock_timer.get.return_value = 0.0
            with (
                patch("ironcore.checkpointing.native.get_logger", return_value=mock_logger),
                patch("ironcore.checkpointing.native.get_timer", return_value=mock_timer),
            ):
                save_checkpoint(config, model, optimizer, scheduler, step=1)

            ckpt_file = os.path.join(tmpdir, "step_1", "pytorch_model.bin")
            assert os.path.exists(ckpt_file)

            # Verify HF config.json was saved
            hf_config_path = os.path.join(tmpdir, "config.json")
            assert os.path.exists(hf_config_path), "HF config.json not saved"

    def test_save_checkpoint_no_hf_config_json(self):
        """When HF fields are None, config.json should NOT be written."""
        from ironcore.checkpointing.native import save_checkpoint

        with tempfile.TemporaryDirectory() as tmpdir:
            config = _make_config(no_save=False, model_path=tmpdir)
            config.model.hf_model_type = None
            config.model.hf_architecture = None

            model = _SimpleModel(config)
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10)

            import logging

            mock_logger = logging.getLogger("test_ckpt")
            mock_timer = MagicMock()
            mock_timer.get.return_value = 0.0
            with (
                patch("ironcore.checkpointing.native.get_logger", return_value=mock_logger),
                patch("ironcore.checkpointing.native.get_timer", return_value=mock_timer),
            ):
                save_checkpoint(config, model, optimizer, scheduler, step=1)

            # config.json should NOT exist when HF fields are missing
            hf_config_path = os.path.join(tmpdir, "config.json")
            assert not os.path.exists(hf_config_path), (
                "config.json should not be written without hf_model_type"
            )

    def test_saved_checkpoint_has_none_hf_config(self):
        """Saved checkpoint dict should have hf_config=None when HF fields not set."""
        from ironcore.checkpointing.native import save_checkpoint

        with tempfile.TemporaryDirectory() as tmpdir:
            config = _make_config(no_save=False, model_path=tmpdir)
            config.model.hf_model_type = None
            config.model.hf_architecture = None

            model = _SimpleModel(config)
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10)

            import logging

            mock_logger = logging.getLogger("test_ckpt")
            mock_timer = MagicMock()
            mock_timer.get.return_value = 0.0
            with (
                patch("ironcore.checkpointing.native.get_logger", return_value=mock_logger),
                patch("ironcore.checkpointing.native.get_timer", return_value=mock_timer),
            ):
                save_checkpoint(config, model, optimizer, scheduler, step=1)

            ckpt_file = os.path.join(tmpdir, "step_1", "pytorch_model.bin")
            ckpt = torch.load(ckpt_file, map_location="cpu", weights_only=False)
            assert "model_state_dict" in ckpt
            assert ckpt["step"] == 1
            assert ckpt.get("hf_config") is None
