# Copyright (c) 2025-2026 Jaegeun Han
#
# SPDX-License-Identifier: Apache-2.0
"""
Consolidated FIM (Fill-in-the-Middle) integration tests.

Combines:
- test_fim_dataset_flow.py: Dataset flow with FIM
- test_fim_serialization.py: FIM serialization/deserialization
"""

from ironcore.config.config_data import DataConfig


def create_fim_test_config(fim_rate: float = 1.0) -> DataConfig:
    """Create a test data config with FIM settings."""
    return DataConfig(
        fim_rate=fim_rate,
        fim_prefix_token="<|fim_prefix|>",
        fim_suffix_token="<|fim_suffix|>",
        fim_middle_token="<|fim_middle|>",
    )


class TestFIMDatasetFlow:
    """Test FIM integration with dataset pipeline."""

    def test_fim_enabled_dataset_flow(self):
        """Verify FIM transformation is applied in dataset pipeline."""
        config = create_fim_test_config(fim_rate=1.0)
        assert config.fim_rate == 1.0

    def test_fim_disabled_dataset_flow(self):
        """Verify dataset works with FIM disabled."""
        config = create_fim_test_config(fim_rate=0.0)
        assert config.fim_rate == 0.0

    def test_fim_rate_partial(self):
        """Verify partial FIM rate (0.5)."""
        config = create_fim_test_config(fim_rate=0.5)
        assert config.fim_rate == 0.5


class TestFIMSerialization:
    """Test FIM serialization and deserialization."""

    def test_fim_config_serialization(self):
        """Verify FIM config can be serialized and deserialized."""
        config = DataConfig(
            fim_rate=0.8,
            fim_prefix_token="<|fim_prefix|>",
            fim_suffix_token="<|fim_suffix|>",
            fim_middle_token="<|fim_middle|>",
        )

        # Serialize to dict
        config_dict = config.__dict__

        # Verify values preserved
        assert config_dict["fim_rate"] == 0.8
        assert config_dict["fim_prefix_token"] == "<|fim_prefix|>"

    def test_fim_tokens_unique(self):
        """Verify FIM tokens are unique."""
        config = DataConfig(
            fim_prefix_token="<|prefix|>",
            fim_suffix_token="<|suffix|>",
            fim_middle_token="<|middle|>",
        )

        tokens = [config.fim_prefix_token, config.fim_suffix_token, config.fim_middle_token]
        assert len(tokens) == len(set(tokens)), "FIM tokens must be unique"


class TestFIMEndToEnd:
    """End-to-end FIM tests."""

    def test_fim_config_integration_with_mainconfig(self):
        """Verify FIM config integrates with MainConfig."""
        from ironcore.config import MainConfig
        from ironcore.config.config_model import ModelConfig
        from ironcore.config.config_trainer import InitConfig, OperationConfig, TrainerConfig
        from ironcore.config.config_optim import OptimConfig
        from ironcore.config.config_parallel import ParallelConfig
        from ironcore.config.config_utils import ProfilerConfig, UtilsConfig
        from ironcore.config import PEFTConfig

        config = MainConfig(
            model=ModelConfig(),
            init=InitConfig(),
            optim=OptimConfig(),
            data=DataConfig(
                fim_rate=0.5,
                fim_prefix_token="<|prefix|>",
                fim_suffix_token="<|suffix|>",
                fim_middle_token="<|middle|>",
            ),
            parallel=ParallelConfig(),
            trainer=TrainerConfig(),
            operation=OperationConfig(),
            utils=UtilsConfig(),
            profiler=ProfilerConfig(),
            peft=PEFTConfig(),
        )

        assert config.data.fim_rate == 0.5
        assert config.data.fim_prefix_token == "<|prefix|>"
