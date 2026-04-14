# Copyright (c) 2025-2026 Jaegeun Han
#
# SPDX-License-Identifier: Apache-2.0
"""Tests for config correctness — regression tests for known bugs."""

import dataclasses
from pathlib import Path

from ironcore.config.config_model import ModelConfig
from ironcore.config.config_trainer import TrainerConfig
from ironcore.config.config_alignment import AlignmentConfig


class TestXPosRemoved:
    """Regression test: XPos was a dead stub that should not exist."""

    def test_xpos_file_does_not_exist(self):
        xpos_path = Path("ironcore/layers/positional_embedding/xpos.py")
        assert not xpos_path.exists(), "xpos.py should have been deleted"


class TestModelConfig:
    """Regression tests for ModelConfig bugs."""

    def test_activation_type_not_duplicated(self):
        """activation_type should appear exactly once (was duplicated at lines 151 and 159)."""
        activation_fields = [
            f for f in dataclasses.fields(ModelConfig) if f.name == "activation_type"
        ]
        assert len(activation_fields) == 1, (
            f"Expected 1 activation_type field, found {len(activation_fields)}"
        )

    def test_activation_type_default_is_gelu(self):
        """activation_type should default to 'gelu'."""
        config = ModelConfig()
        assert config.activation_type == "gelu"


class TestTrainerConfig:
    """Regression tests for TrainerConfig bugs."""

    def test_no_pipeline_model_parallel_size(self):
        """pipeline_model_parallel_size should not exist (removed as unimplemented ghost)."""
        assert not hasattr(TrainerConfig(), "pipeline_model_parallel_size"), (
            "pipeline_model_parallel_size should have been removed from TrainerConfig"
        )


class TestAlignmentConfig:
    """Regression tests for AlignmentConfig fields."""

    def test_offload_ref_model_field_exists(self):
        """offload_ref_model should be a declared field defaulting to False."""
        config = AlignmentConfig()
        assert hasattr(config, "offload_ref_model")
        assert config.offload_ref_model is False

    def test_offload_ref_model_is_configurable(self):
        """offload_ref_model should be settable."""
        config = AlignmentConfig(offload_ref_model=True)
        assert config.offload_ref_model is True
