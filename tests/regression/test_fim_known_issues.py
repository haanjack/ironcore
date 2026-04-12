# Copyright (c) 2025-2026 Jaegeun Han
#
# SPDX-License-Identifier: Apache-2.0
"""
Regression tests for FIM known issues.

Tests that prevent regression of specific identified bugs and issues.
"""

from random import Random

import pytest
from tests.fixtures.utils import verify_psm_structure

from ironcore.dataloader.data_config import DatasetConfig, UniversalDataConfig
from ironcore.preprocessing.serializer import DataSerializer


class TestDefaultInconsistencyRegression:
    """Regression tests for default value inconsistencies."""

    def test_regression_fim_rate_consistency(self):
        """
        REGRESSION: Verify that dataclass and parser defaults match.

        FIM config is at DataConfig level (not per-dataset).
        """
        # Direct instantiation
        config_direct = UniversalDataConfig(
            datasets=[DatasetConfig(name="test", source="dummy", task_type="pretrain")]
        )

        # YAML parsing
        config_dict = {
            "train_datasets": [{"name": "test", "dataset_path": "dummy", "task_type": "pretrain"}],
            "vocab_name_or_path": "gpt2",
        }
        config_from_yaml = UniversalDataConfig._parse_config_dict(config_dict)

        # CONSISTENT: Default value should be the same
        assert config_direct.fim_rate == config_from_yaml.fim_rate


class TestMissingTokenErrorRegression:
    """Regression tests for missing FIM token errors."""

    def test_regression_missing_fim_tokens_error(
        self, test_tokenizer_without_fim, test_config_fim_enabled
    ):
        """
        REGRESSION: Clear error when tokens not in vocab.

        Missing FIM tokens should raise ValueError with helpful message.
        """
        serializer = DataSerializer(
            test_config_fim_enabled, test_tokenizer_without_fim, verbose=False
        )

        with pytest.raises(ValueError) as exc_info:
            serializer._get_token_id("<fim_prefix>")

        error_msg = str(exc_info.value)

        # Should contain helpful information
        assert "not found in tokenizer vocabulary" in error_msg
        assert "add_special_tokens" in error_msg

    def test_regression_error_message_quality(
        self, test_tokenizer_without_fim, test_config_fim_enabled
    ):
        """
        REGRESSION: Error message provides actionable instructions.

        Error should tell user how to fix the problem.
        """
        serializer = DataSerializer(
            test_config_fim_enabled, test_tokenizer_without_fim, verbose=False
        )

        with pytest.raises(ValueError) as exc_info:
            serializer._get_token_id("<fim_prefix>")

        error_msg = str(exc_info.value)

        # Check for actionable instructions
        assert "tokenizer.add_special_tokens" in error_msg or "add_special_tokens" in error_msg
        assert "save_pretrained" in error_msg


class TestShortSequenceRegression:
    """Regression tests for short sequence handling."""

    def test_regression_short_sequence_untransformed(self, serializer_with_fim, fim_token_ids):
        """
        REGRESSION: Sequences < 10 tokens are not transformed.

        Code checks `if length < 10: return token_ids` (line 555-556).
        """
        # Test boundary: 9 tokens (should not transform)
        token_ids = list(range(9))
        rng = Random(42)

        transformed = serializer_with_fim._apply_fim_transformation(
            token_ids,
            fim_token_ids["prefix"],
            fim_token_ids["suffix"],
            fim_token_ids["middle"],
            rng,
        )

        assert transformed == token_ids  # Unchanged

    def test_regression_boundary_sequence_transformed(self, serializer_with_fim, fim_token_ids):
        """
        REGRESSION: Sequences >= 10 tokens ARE transformed.

        Boundary is at 10 tokens.
        """
        # Test boundary: 10 tokens (should transform)
        token_ids = list(range(10))
        rng = Random(42)

        transformed = serializer_with_fim._apply_fim_transformation(
            token_ids,
            fim_token_ids["prefix"],
            fim_token_ids["suffix"],
            fim_token_ids["middle"],
            rng,
        )

        assert transformed != token_ids  # Changed
        assert verify_psm_structure(
            transformed, fim_token_ids["prefix"], fim_token_ids["suffix"], fim_token_ids["middle"]
        )


class TestEmptyMiddleRegression:
    """Regression tests for empty middle section handling."""

    def test_regression_empty_middle_handled(self, serializer_with_fim, fim_token_ids):
        """
        REGRESSION: Empty middle is prevented.

        Code has edge case handling (line 571-573):
        ```
        if not middle:
            middle = [token_ids[split1]]
        ```

        However, this shouldn't happen with random.sample() which guarantees
        different split points. This test verifies the safeguard works if needed.
        """
        # This is difficult to trigger with random.sample()
        # The test documents that empty middle is handled

        # Try many seeds to see if we can trigger edge case
        for seed in range(100):
            token_ids = list(range(10, 30))
            rng = Random(seed)

            transformed = serializer_with_fim._apply_fim_transformation(
                token_ids,
                fim_token_ids["prefix"],
                fim_token_ids["suffix"],
                fim_token_ids["middle"],
                rng,
            )

            # Extract middle section
            transformed.index(fim_token_ids["prefix"])
            transformed.index(fim_token_ids["suffix"])
            fm_pos = transformed.index(fim_token_ids["middle"])

            middle_section = transformed[fm_pos + 1 :]

            # Should never be empty
            assert len(middle_section) > 0, f"Empty middle with seed {seed}"


class TestMetadataTypeRegression:
    """Regression tests for metadata type tracking."""

    def test_regression_no_type_tracking(self, temp_dir, test_tokenizer_with_fim):
        """
        REGRESSION: No metadata distinction between FIM and non-FIM.

        Currently metadata type="pretrain" for both FIM and non-FIM sequences.
        Cannot distinguish them from metadata alone.

        Future enhancement: Add "pretrain_fim" type for FIM sequences.
        """
        import numpy as np
        from tests.fixtures.mocks import MockDataset

        config = UniversalDataConfig(
            datasets=[
                DatasetConfig(
                    name="test",
                    source="dummy",
                    task_type="pretrain",
                    text_column="text",
                )
            ],
            fim_rate=0.5,  # Mix of FIM and non-FIM (FIM config is at DataConfig level)
            fim_prefix_token="<fim_prefix>",
            fim_suffix_token="<fim_suffix>",
            fim_middle_token="<fim_middle>",
            vocab_name_or_path="gpt2",
            seq_length=1024,
            preprocessed_dir=temp_dir / "preprocessed",
            cache_dir=temp_dir / "cache",
        )

        serializer = DataSerializer(config, test_tokenizer_with_fim, verbose=False)

        dataset = MockDataset([{"text": f"def function_{i}():\n    return {i}"} for i in range(20)])

        output_path = config.get_dataset_output_path(config.datasets[0])
        bin_path = output_path / "data.bin"
        idx_path = output_path / "data.idx"

        serializer._serialize_pretrain(dataset, config.datasets[0], bin_path, idx_path)

        # Load metadata
        metadata = np.load(output_path / "data.idx.npy", allow_pickle=True)

        # All have type="pretrain", no distinction
        for meta in metadata:
            assert meta["type"] == "pretrain"

        # Cannot tell which are FIM from metadata alone
        # Would need to check tokens for FIM special tokens


class TestTiktokenRegression:
    """Regression tests for tiktoken limitations."""

    def test_regression_tiktoken_unsupported(self):
        """
        REGRESSION: Document that tiktoken doesn't support FIM tokens well.

        tiktoken tokenizers cannot easily add new special tokens.
        FIM requires HuggingFace tokenizers.

        This is documented in error messages from _get_token_id.
        """
        # Tiktoken limitation is documented
        # FIM functionality requires HuggingFace tokenizers
        # This test serves as documentation
        pass

    def test_regression_hf_tokenizer_required(self, test_config_fim_enabled):
        """
        REGRESSION: FIM requires HuggingFace tokenizer.

        Non-HF tokenizers raise TypeError with helpful message.
        """

        class FakeTiktoken:
            """Mock tiktoken-like tokenizer."""

            def __init__(self):
                self.eos_token_id = 0

            def encode(self, text):
                return [1, 2, 3]

        fake_tokenizer = FakeTiktoken()
        serializer = DataSerializer(test_config_fim_enabled, fake_tokenizer, verbose=False)

        with pytest.raises(TypeError) as exc_info:
            serializer._get_token_id("<fim_prefix>")

        error_msg = str(exc_info.value)
        assert "Unsupported tokenizer type" in error_msg
        assert "HuggingFace" in error_msg


class TestRandomSampleRegression:
    """Regression tests for random.sample() usage."""

    def test_regression_sample_guarantees_different_values(
        self, serializer_with_fim, fim_token_ids
    ):
        """
        REGRESSION: random.sample() guarantees different split points.

        Code uses `rng.sample(range(1, length), 2)` which guarantees
        split1 != split2. The double-check (line 563-564) is redundant
        but serves as defensive programming.
        """
        token_ids = list(range(50))

        # Test many times to verify sample always produces different values
        for seed in range(50):
            rng = Random(seed)
            transformed = serializer_with_fim._apply_fim_transformation(
                token_ids.copy(),
                fim_token_ids["prefix"],
                fim_token_ids["suffix"],
                fim_token_ids["middle"],
                rng,
            )

            # All sections should be non-empty (verifies split1 != split2)
            fp_pos = transformed.index(fim_token_ids["prefix"])
            fs_pos = transformed.index(fim_token_ids["suffix"])
            fm_pos = transformed.index(fim_token_ids["middle"])

            prefix_len = fs_pos - fp_pos - 1
            suffix_len = fm_pos - fs_pos - 1
            middle_len = len(transformed) - fm_pos - 1

            assert prefix_len > 0
            assert middle_len > 0  # Would be 0 if split1 == split2
            assert suffix_len > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
