"""
Unit tests for FIM configuration.

Tests UniversalDataConfig and ModelConfig for FIM-specific fields.
FIM tokens are now in ModelConfig, FIM behavior (rate, split_type) in UniversalDataConfig.
"""

from ironcore.dataloader.data_config import DatasetConfig, UniversalDataConfig


class TestFIMConfigFields:
    """Test FIM configuration fields are present in UniversalDataConfig."""

    def test_fim_config_fields_present(self):
        """Verify UniversalDataConfig has all FIM fields."""
        print(f"DEBUG: UniversalDataConfig class: {UniversalDataConfig}")
        config = UniversalDataConfig(
            datasets=[DatasetConfig(name="test", source="dummy", task_type="pretrain")]
        )
        print(f"DEBUG: config instance: {config}")
        print(f"DEBUG: config fields: {config.__dataclass_fields__.keys()}")
        assert hasattr(config, "fim_rate")
        assert hasattr(config, "fim_split_type")
        assert hasattr(config, "fim_prefix_token")
        assert hasattr(config, "fim_suffix_token")
        assert hasattr(config, "fim_middle_token")

    def test_fim_rate_default(self):
        """Verify fim_rate defaults to 0.0 (disabled)."""
        config = UniversalDataConfig(
            datasets=[DatasetConfig(name="test", source="dummy", task_type="pretrain")]
        )
        assert config.fim_rate == 0.0

    def test_fim_special_token_defaults(self):
        """Verify FIM special tokens have correct default values."""
        config = UniversalDataConfig(
            datasets=[DatasetConfig(name="test", source="dummy", task_type="pretrain")]
        )

        assert config.fim_prefix_token == "<fim_prefix>"
        assert config.fim_suffix_token == "<fim_suffix>"
        assert config.fim_middle_token == "<fim_middle>"

    def test_fim_split_type_default(self):
        """Verify fim_split_type defaults to 'random'."""
        config = UniversalDataConfig(
            datasets=[DatasetConfig(name="test", source="dummy", task_type="pretrain")]
        )

        assert config.fim_split_type == "random"


class TestFIMConfigValidation:
    """Test validation logic for FIM-specific fields."""

    def test_fim_rate_valid_range(self):
        """Test valid fim_rate values (0.0 to 1.0)."""
        for rate in [0.0, 0.25, 0.5, 0.75, 1.0]:
            config = UniversalDataConfig(
                datasets=[DatasetConfig(name="test", source="dummy", task_type="pretrain")],
                fim_rate=rate,
            )
            assert config.fim_rate == rate

    def test_fim_split_type_valid_values(self):
        """Test valid fim_split_type values."""
        for split_type in ["random", "line_aware"]:
            config = UniversalDataConfig(
                datasets=[DatasetConfig(name="test", source="dummy", task_type="pretrain")],
                fim_split_type=split_type,
            )
            assert config.fim_split_type == split_type


class TestFIMYAMLParsing:
    """Test parsing FIM configuration from YAML."""

    def test_fim_yaml_parsing_basic(self, tmp_path):
        """Test parsing FIM config from YAML (top-level)."""
        yaml_content = """
        train_datasets:
          - name: test_fim
            dataset_path: dummy
            task_type: pretrain

        vocab_name_or_path: gpt2
        seq_length: 1024

        # FIM settings at top level
        fim_rate: 0.7
        fim_prefix_token: "<fim_prefix>"
        fim_suffix_token: "<fim_suffix>"
        fim_middle_token: "<fim_middle>"
        """

        yaml_path = tmp_path / "config.yaml"
        with open(yaml_path, "w") as f:
            f.write(yaml_content)

        config = UniversalDataConfig.from_yaml(yaml_path)

        assert len(config.datasets) == 1
        assert config.fim_rate == 0.7
        assert config.fim_prefix_token == "<fim_prefix>"
        assert config.fim_suffix_token == "<fim_suffix>"
        assert config.fim_middle_token == "<fim_middle>"

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

        # Default fim_rate is 0.0 (disabled)
        assert config.fim_rate == 0.0

    def test_fim_yaml_custom_tokens(self, tmp_path):
        """Test parsing custom FIM token names."""
        yaml_content = """
        train_datasets:
          - name: test
            dataset_path: dummy
            task_type: pretrain

        vocab_name_or_path: gpt2
        seq_length: 1024

        fim_rate: 0.5
        fim_prefix_token: "[PREFIX]"
        fim_suffix_token: "[SUFFIX]"
        fim_middle_token: "[MIDDLE]"
        """

        yaml_path = tmp_path / "config.yaml"
        with open(yaml_path, "w") as f:
            f.write(yaml_content)

        config = UniversalDataConfig.from_yaml(yaml_path)

        assert config.fim_prefix_token == "[PREFIX]"
        assert config.fim_suffix_token == "[SUFFIX]"
        assert config.fim_middle_token == "[MIDDLE]"


class TestFIMConfigConsistency:
    """Test consistency across different ways of setting FIM config."""

    def test_default_consistency_dataclass_vs_yaml_parser(self):
        """Verify that dataclass and parser defaults match."""
        # Direct instantiation: uses dataclass default
        config_direct = UniversalDataConfig(
            datasets=[DatasetConfig(name="test", source="dummy", task_type="pretrain")]
        )
        assert config_direct.fim_rate == 0.0

        # Empty YAML parsing: uses _parse_config_dict defaults
        config_yaml = UniversalDataConfig._parse_config_dict(
            {
                "train_datasets": [
                    {"name": "test", "dataset_path": "dummy", "task_type": "pretrain"}
                ]
            }
        )
        assert config_yaml.fim_rate == 0.0

        assert config_direct.fim_rate == config_yaml.fim_rate
        assert config_direct.fim_prefix_token == config_yaml.fim_prefix_token
