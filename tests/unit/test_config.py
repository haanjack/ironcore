# Copyright (c) 2025-2026 Jaegeun Han
#
# SPDX-License-Identifier: Apache-2.0
"""Tests for config correctness — regression tests for known bugs."""

import dataclasses
from pathlib import Path

from ironcore.config.config_alignment import AlignmentConfig
from ironcore.config.config_model import ModelConfig
from ironcore.config.config_trainer import TrainerConfig


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


class TestInlineDataBlockTypoGuard:
    """A misspelled key in an inline `data:` block must not be accepted.

    Every other config group is checked against its dataclass fields, but the
    data path guarded with `sub_group_key not in config_group` — a key drawn
    from that same dict, so the check never fired. A typo was setattr'd as a
    stray attribute while the real field silently kept its default.
    """

    @staticmethod
    def _apply(block):
        from ironcore.config import _update_data_config_from_yaml
        from ironcore.config.config_data import DataConfig

        holder = type("Holder", (), {})()
        holder.data = DataConfig()
        _update_data_config_from_yaml(holder, "data", block)
        return holder.data

    def test_known_key_is_applied(self):
        data = self._apply({"seq_length": 256})
        assert data.seq_length == 256

    def test_misspelled_key_raises(self):
        import pytest

        with pytest.raises(ValueError, match="seq_lenght"):
            self._apply({"seq_lenght": 256})

    def test_misspelled_key_does_not_reach_the_object(self):
        import pytest

        from ironcore.config.config_data import DataConfig

        default = DataConfig().seq_length
        with pytest.raises(ValueError):
            self._apply({"seq_lenght": 999999})
        assert DataConfig().seq_length == default
