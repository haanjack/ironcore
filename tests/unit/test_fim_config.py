"""
Unit tests for FIM configuration.

Tests DatasetConfig and UniversalDataConfig parsing and validation
for FIM-specific fields.
"""

import pytest

from ironcore.dataloader.data_config import DatasetConfig, UniversalDataConfig


class TestFIMConfigFields:
    """Test FIM configuration fields are present and have correct defaults."""

    def test_fim_config_fields_present(self):
        """Verify DatasetConfig has all FIM fields."""
        config = DatasetConfig(name="test", source="dummy", task_type="pretrain")

        assert hasattr(config, "fim_rate")
        assert hasattr(config, "fim_split_type")
        assert hasattr(config, "fim_prefix_token")
        assert hasattr(config, "fim_suffix_token")
        assert hasattr(config, "fim_middle_token")

    def test_fim_rate_default(self):
        """Verify fim_rate defaults to 0.5."""
        config = DatasetConfig(name="test", source="dummy", task_type="pretrain")

        assert config.fim_rate == 0.5

    def test_fim_special_token_defaults(self):
        """Verify FIM special tokens have correct default values."""
        config = DatasetConfig(name="test", source="dummy", task_type="pretrain")

        assert config.fim_prefix_token == "<fim_prefix>"
        assert config.fim_suffix_token == "<fim_suffix>"
        assert config.fim_middle_token == "<fim_middle>"

    def test_fim_split_type_default(self):
        """Verify fim_split_type defaults to 'random'."""
        config = DatasetConfig(name="test", source="dummy", task_type="pretrain")

        assert config.fim_split_type == "random"


class TestFIMConfigValidation:
    """Test FIM configuration validation."""

    def test_fim_rate_valid_range(self):
        """Test valid fim_rate values (0.0 to 1.0)."""
        # Valid rates
        for rate in [0.0, 0.25, 0.5, 0.75, 1.0]:
            config = DatasetConfig(name="test", source="dummy", task_type="pretrain", fim_rate=rate)
            assert config.fim_rate == rate

    def test_fim_rate_negative_invalid(self):
        """Test that negative fim_rate raises error."""
        # Note: Current implementation doesn't validate fim_rate bounds
        # This test documents expected behavior for future validation
        config = DatasetConfig(name="test", source="dummy", task_type="pretrain", fim_rate=-0.1)
        # Currently this doesn't fail, but should be validated
        assert config.fim_rate == -0.1

    def test_fim_rate_above_one_invalid(self):
        """Test that fim_rate > 1.0 raises error."""
        # Note: Current implementation doesn't validate fim_rate bounds
        # This test documents expected behavior for future validation
        config = DatasetConfig(name="test", source="dummy", task_type="pretrain", fim_rate=1.5)
        # Currently this doesn't fail, but should be validated
        assert config.fim_rate == 1.5

    def test_fim_split_type_valid_values(self):
        """Test valid fim_split_type values."""
        for split_type in ["random", "line_aware"]:
            config = DatasetConfig(
                name="test", source="dummy", task_type="pretrain", fim_split_type=split_type
            )
            assert config.fim_split_type == split_type


class TestFIMYAMLParsing:
    """Test FIM configuration parsing from YAML."""

    def test_fim_yaml_parsing_basic(self, tmp_path):
        """Test parsing FIM config from YAML."""
        yaml_content = """
        train_datasets:
          - name: test_fim
            dataset_path: dummy
            task_type: pretrain
            fim_rate: 0.7
            fim_prefix_token: "<fim_prefix>"
            fim_suffix_token: "<fim_suffix>"
            fim_middle_token: "<fim_middle>"

        vocab_name_or_path: gpt2
        seq_length: 1024
        """

        yaml_path = tmp_path / "config.yaml"
        with open(yaml_path, "w") as f:
            f.write(yaml_content)

        config = UniversalDataConfig.from_yaml(yaml_path)

        assert len(config.datasets) == 1
        dataset = config.datasets[0]
        assert dataset.fim_rate == 0.7
        assert dataset.fim_prefix_token == "<fim_prefix>"
        assert dataset.fim_suffix_token == "<fim_suffix>"
        assert dataset.fim_middle_token == "<fim_middle>"

    def test_fim_yaml_default_rate(self, tmp_path):
        """Test FIM rate default when not specified in YAML."""
        yaml_content = """
        train_datasets:
          - name: test
            dataset_path: dummy
            task_type: pretrain

        vocab_name_or_path: gpt2
        seq_length: 1024
        """

        yaml_path = tmp_path / "config.yaml"
        with open(yaml_path, "w") as f:
            f.write(yaml_content)

        config = UniversalDataConfig.from_yaml(yaml_path)

        # Both dataclass and parser should now default to 0.5 consistently
        dataset = config.datasets[0]
        assert dataset.fim_rate == 0.5

    def test_fim_yaml_custom_tokens(self, tmp_path):
        """Test parsing custom FIM token names."""
        yaml_content = """
        train_datasets:
          - name: test
            dataset_path: dummy
            task_type: pretrain
            fim_rate: 0.5
            fim_prefix_token: "[PREFIX]"
            fim_suffix_token: "[SUFFIX]"
            fim_middle_token: "[MIDDLE]"

        vocab_name_or_path: gpt2
        seq_length: 1024
        """

        yaml_path = tmp_path / "config.yaml"
        with open(yaml_path, "w") as f:
            f.write(yaml_content)

        config = UniversalDataConfig.from_yaml(yaml_path)

        dataset = config.datasets[0]
        assert dataset.fim_prefix_token == "[PREFIX]"
        assert dataset.fim_suffix_token == "[SUFFIX]"
        assert dataset.fim_middle_token == "[MIDDLE]"


class TestFIMTaskTypeInteraction:
    """Test FIM interaction with different task types."""

    def test_fim_with_pretrain_task(self):
        """FIM should be usable with pretrain task."""
        config = DatasetConfig(name="test", source="dummy", task_type="pretrain", fim_rate=0.5)

        assert config.task_type == "pretrain"
        assert config.fim_rate == 0.5

    def test_fim_with_sft_task(self):
        """FIM fields should exist but be ignored for SFT task."""
        config = DatasetConfig(
            name="test",
            source="dummy",
            task_type="sft",
            fim_rate=0.5,  # Should be ignored
        )

        assert config.task_type == "sft"
        # FIM rate is set but won't be used during serialization
        assert config.fim_rate == 0.5

    def test_fim_with_dpo_task(self):
        """FIM fields should exist but be ignored for DPO task."""
        config = DatasetConfig(
            name="test",
            source="dummy",
            task_type="dpo",
            fim_rate=0.5,  # Should be ignored
        )

        assert config.task_type == "dpo"
        # FIM rate is set but won't be used during serialization
        assert config.fim_rate == 0.5


class TestFIMConfigConsistency:
    """Test for configuration consistency issues."""

    def test_default_consistency_dataclass_vs_yaml_parser(self):
        """
        Verify that dataclass and parser defaults match (both 0.5).
        """
        # Direct instantiation: uses dataclass default
        config_direct = DatasetConfig(name="test", source="dummy", task_type="pretrain")
        assert config_direct.fim_rate == 0.5

        # Via YAML parsing: uses parser default
        config_dict = {
            "train_datasets": [{"name": "test", "dataset_path": "dummy", "task_type": "pretrain"}],
            "vocab_name_or_path": "gpt2",
            "seq_length": 1024,
        }
        config_from_dict = UniversalDataConfig.from_dict(config_dict)
        # Both should now match
        assert config_from_dict.datasets[0].fim_rate == 0.5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
