# Copyright (c) 2025-2026 Jaegeun Han
#
# SPDX-License-Identifier: Apache-2.0

"""
Unit tests for OffloadConfig.

Tests cover:
1. Default values (all disabled)
2. Validation rules (optimizer_offload requires enabled, etc.)
3. activation spill validation (mutual exclusion with activation_recompute)
4. Integration with MainConfig
"""

import warnings

import pytest

from ironcore.offload.config import OffloadConfig


class TestOffloadConfigDefaults:
    """Test that all offload features default to disabled."""

    def test_defaults_all_disabled(self):
        config = OffloadConfig()
        assert config.enabled is False
        assert config.optimizer_offload is False
        assert config.weight_offload is False
        assert config.activation_spill is False

    def test_default_precision(self):
        config = OffloadConfig()
        assert config.optimizer_state_precision == "fp32"

    def test_default_min_param_elements(self):
        config = OffloadConfig()
        assert config.optimizer_min_param_elements == 65536

    def test_default_pinned_pool_size(self):
        config = OffloadConfig()
        assert config.pinned_memory_pool_gb == 100.0

    def test_default_spill_granularity(self):
        config = OffloadConfig()
        assert config.activation_spill_granularity == "sub_layer"


class TestOffloadConfigValidation:
    """Test config validation rules."""

    def test_optimizer_offload_requires_enabled(self):
        """optimizer_offload=True without enabled=True should raise."""
        from ironcore.config import _config_validation

        config = _make_minimal_main_config()
        config.offload.optimizer_offload = True
        # enabled defaults to False
        with pytest.raises(ValueError, match="requires offload.enabled"):
            _config_validation(config)

    def test_invalid_precision_raises(self):
        """Invalid optimizer_state_precision should raise."""
        from ironcore.config import _config_validation

        config = _make_minimal_main_config()
        config.offload.enabled = True
        config.offload.optimizer_offload = True
        config.offload.optimizer_state_precision = "fp8"
        with pytest.raises(ValueError, match="must be fp32, fp16, or bf16"):
            _config_validation(config)

    def test_valid_precisions_accepted(self):
        """fp32, fp16, bf16 should all pass validation."""
        from ironcore.config import _config_validation

        for precision in ("fp32", "fp16", "bf16"):
            config = _make_minimal_main_config()
            config.offload.enabled = True
            config.offload.optimizer_offload = True
            config.offload.optimizer_state_precision = precision
            # Should not raise
            _config_validation(config)

    def test_activation_spill_requires_enabled(self):
        """activation_spill=True without enabled=True should raise."""
        from ironcore.config import _config_validation

        config = _make_minimal_main_config()
        config.offload.activation_spill = True
        # enabled defaults to False
        with pytest.raises(ValueError, match="requires offload.enabled"):
            _config_validation(config)

    def test_invalid_spill_granularity_raises(self):
        """Invalid activation_spill_granularity should raise."""
        from ironcore.config import _config_validation

        config = _make_minimal_main_config()
        config.offload.enabled = True
        config.offload.activation_spill = True
        config.offload.activation_spill_granularity = "per_token"
        with pytest.raises(ValueError, match="must be 'sub_layer' or 'full_layer'"):
            _config_validation(config)

    def test_activation_spill_disables_recompute(self):
        """activation_spill=true auto-disables activation_recompute with warning."""
        from ironcore.config import _config_validation

        config = _make_minimal_main_config()
        config.offload.enabled = True
        config.offload.activation_spill = True
        config.operation.activation_recompute = True

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            _config_validation(config)

        assert config.operation.activation_recompute is False
        assert len(w) == 1
        assert "Disabling activation_recompute" in str(w[0].message)

    def test_all_enabled_valid(self):
        """Enabling everything validly should not raise."""
        from ironcore.config import _config_validation

        config = _make_minimal_main_config()
        config.offload.enabled = True
        config.offload.optimizer_offload = True
        config.offload.activation_spill = True
        # Should not raise
        _config_validation(config)


class TestOffloadConfigUpdate:
    """Test config update via __call__ (BaseConfig pattern)."""

    def test_update_single_field(self):
        config = OffloadConfig()
        config(enabled=True)
        assert config.enabled is True
        assert config.optimizer_offload is False  # unchanged

    def test_update_multiple_fields(self):
        config = OffloadConfig()
        config(enabled=True, optimizer_offload=True, optimizer_state_precision="bf16")
        assert config.enabled is True
        assert config.optimizer_offload is True
        assert config.optimizer_state_precision == "bf16"


def _make_minimal_main_config():
    """Create a minimal MainConfig with valid required fields for testing."""
    from ironcore.config import MainConfig

    config = MainConfig(
        model=__import__("ironcore.config.config_model", fromlist=["ModelConfig"]).ModelConfig(),
        init=__import__("ironcore.config.config_trainer", fromlist=["InitConfig"]).InitConfig(),
        optim=__import__("ironcore.config.config_optim", fromlist=["OptimConfig"]).OptimConfig(),
        data=__import__("ironcore.config.config_data", fromlist=["DataConfig"]).DataConfig(),
        parallel=__import__(
            "ironcore.config.config_parallel", fromlist=["ParallelConfig"]
        ).ParallelConfig(),
        trainer=__import__(
            "ironcore.config.config_trainer", fromlist=["TrainerConfig"]
        ).TrainerConfig(),
        operation=__import__(
            "ironcore.config.config_trainer", fromlist=["OperationConfig"]
        ).OperationConfig(
            train_steps=100,
        ),
        utils=__import__("ironcore.config.config_utils", fromlist=["UtilsConfig"]).UtilsConfig(),
        profiler=__import__(
            "ironcore.config.config_utils", fromlist=["ProfilerConfig"]
        ).ProfilerConfig(),
        peft=__import__("ironcore.config.config_peft", fromlist=["PEFTConfig"]).PEFTConfig(),
        alignment=__import__(
            "ironcore.config.config_alignment", fromlist=["AlignmentConfig"]
        ).AlignmentConfig(),
        offload=OffloadConfig(),
    )
    # Set required fields for validation
    config.trainer.micro_batch_size = 4
    config.trainer.train_batch_size = 4
    config.trainer.gradient_accumulation_steps = 1
    config.parallel.world_size = 1
    return config
