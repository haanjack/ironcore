# Copyright (c) 2025-2026 Jaegeun Han
#
# SPDX-License-Identifier: Apache-2.0
"""
Integration tests for FIM dataset flow.

Tests the full pipeline: Config → Serialize → Dataset → Collator → Batch.
"""

import numpy as np
import pytest
from tests.fixtures.mocks import MockDataset
from tests.fixtures.utils import verify_psm_structure

from ironcore.dataloader.data_config import DatasetConfig, UniversalDataConfig
from ironcore.preprocessing.serializer import DataSerializer


class TestFIMFullPipeline:
    """Test full FIM pipeline from config to batch."""

    def test_fim_full_pipeline_basic(self, temp_dir, test_tokenizer_with_fim, fim_token_ids):
        """Test complete pipeline: Config → Serialize → verify."""
        # Create config
        config = UniversalDataConfig(
            datasets=[
                DatasetConfig(
                    name="test_fim_pipeline",
                    source="dummy",
                    task_type="pretrain",
                    fim_rate=1.0,  # 100% for deterministic testing
                    fim_prefix_token="<fim_prefix>",
                    fim_suffix_token="<fim_suffix>",
                    fim_middle_token="<fim_middle>",
                    text_column="text",
                )
            ],
            vocab_name_or_path="gpt2",
            seq_length=512,
            preprocessed_dir=temp_dir / "preprocessed",
            cache_dir=temp_dir / "cache",
        )

        # Create serializer
        serializer = DataSerializer(config, test_tokenizer_with_fim, verbose=False)

        # Create dataset
        dataset = MockDataset(
            [
                {"text": "def hello():\n    print('world')\n    return 42"},
                {"text": "class Foo:\n    def __init__(self):\n        self.x = 10"},
                {"text": "for i in range(10):\n    print(i)"},
            ]
        )

        # Serialize
        output_path = config.get_dataset_output_path(config.datasets[0])
        bin_path = output_path / "data.bin"
        idx_path = output_path / "data.idx"

        serializer._serialize_pretrain(dataset, config.datasets[0], bin_path, idx_path)

        # Load and verify
        tokens = np.fromfile(bin_path, dtype=np.uint16)
        metadata = np.load(output_path / "data.idx.npy", allow_pickle=True)

        assert len(metadata) == 3

        # Verify all samples have FIM structure
        for meta in metadata:
            offset = meta["offset"]
            length = meta["length"]
            sample_tokens = tokens[offset : offset + length].tolist()

            # Remove EOS
            sample_tokens_no_eos = sample_tokens[:-1]

            # Should have PSM structure
            assert verify_psm_structure(
                sample_tokens_no_eos,
                fim_token_ids["prefix"],
                fim_token_ids["suffix"],
                fim_token_ids["middle"],
            )

    def test_fim_serialization_output_paths(self, temp_dir, test_tokenizer_with_fim):
        """Test that serialization creates correct output paths."""
        config = UniversalDataConfig(
            datasets=[
                DatasetConfig(
                    name="test_paths",
                    source="dummy",
                    task_type="pretrain",
                    fim_rate=0.5,
                    fim_prefix_token="<fim_prefix>",
                    fim_suffix_token="<fim_suffix>",
                    fim_middle_token="<fim_middle>",
                    text_column="text",
                )
            ],
            vocab_name_or_path="gpt2",
            seq_length=512,
            preprocessed_dir=temp_dir / "preprocessed",
            cache_dir=temp_dir / "cache",
        )

        serializer = DataSerializer(config, test_tokenizer_with_fim, verbose=False)

        dataset = MockDataset(
            [
                {"text": "Sample text for path testing."},
            ]
        )

        output_path = config.get_dataset_output_path(config.datasets[0])
        bin_path = output_path / "data.bin"
        idx_path = output_path / "data.idx"

        serializer._serialize_pretrain(dataset, config.datasets[0], bin_path, idx_path)

        # Verify paths (np.save creates .npy extension)
        expected_base = temp_dir / "preprocessed" / "test_paths" / "pretrain"
        assert bin_path == expected_base / "data.bin"
        assert idx_path == expected_base / "data.idx"
        assert bin_path.exists()
        assert (expected_base / "data.idx.npy").exists()


class TestFIMTokensInData:
    """Test that FIM tokens appear correctly in serialized data."""

    def test_fim_tokens_in_batch(self, temp_dir, test_tokenizer_with_fim, fim_token_ids):
        """Test FIM special tokens are present in batch data."""
        config = UniversalDataConfig(
            datasets=[
                DatasetConfig(
                    name="test_batch",
                    source="dummy",
                    task_type="pretrain",
                    fim_rate=1.0,  # All sequences
                    fim_prefix_token="<fim_prefix>",
                    fim_suffix_token="<fim_suffix>",
                    fim_middle_token="<fim_middle>",
                    text_column="text",
                )
            ],
            vocab_name_or_path="gpt2",
            seq_length=512,
            preprocessed_dir=temp_dir / "preprocessed",
            cache_dir=temp_dir / "cache",
        )

        serializer = DataSerializer(config, test_tokenizer_with_fim, verbose=False)

        dataset = MockDataset(
            [
                {"text": "def function():\n    x = 1\n    y = 2\n    z = 3\n    return x + y + z"},
            ]
        )

        output_path = config.get_dataset_output_path(config.datasets[0])
        bin_path = output_path / "data.bin"
        idx_path = output_path / "data.idx"

        serializer._serialize_pretrain(dataset, config.datasets[0], bin_path, idx_path)

        # Load tokens and metadata (np.save creates .npy extension)
        tokens = np.fromfile(bin_path, dtype=np.uint16)
        metadata = np.load(output_path / "data.idx.npy", allow_pickle=True)

        # Get first sample tokens
        offset = metadata[0]["offset"]
        length = metadata[0]["length"]
        sample_tokens = tokens[offset : offset + length].tolist()

        # Verify FIM tokens present in sample
        assert fim_token_ids["prefix"] in sample_tokens
        assert fim_token_ids["suffix"] in sample_tokens
        assert fim_token_ids["middle"] in sample_tokens

    def test_fim_special_token_positions(self, temp_dir, test_tokenizer_with_fim, fim_token_ids):
        """Test that FIM special tokens are in correct positions."""
        config = UniversalDataConfig(
            datasets=[
                DatasetConfig(
                    name="test_positions",
                    source="dummy",
                    task_type="pretrain",
                    fim_rate=1.0,
                    fim_prefix_token="<fim_prefix>",
                    fim_suffix_token="<fim_suffix>",
                    fim_middle_token="<fim_middle>",
                    text_column="text",
                )
            ],
            vocab_name_or_path="gpt2",
            seq_length=512,
            preprocessed_dir=temp_dir / "preprocessed",
            cache_dir=temp_dir / "cache",
        )

        serializer = DataSerializer(config, test_tokenizer_with_fim, verbose=False)

        dataset = MockDataset(
            [
                {"text": "def function():\n    x = 1\n    y = 2\n    z = 3\n    return x + y + z"},
            ]
        )

        output_path = config.get_dataset_output_path(config.datasets[0])
        bin_path = output_path / "data.bin"
        idx_path = output_path / "data.idx"

        serializer._serialize_pretrain(dataset, config.datasets[0], bin_path, idx_path)

        # Load tokens
        tokens = np.fromfile(bin_path, dtype=np.uint16)
        metadata = np.load(output_path / "data.idx.npy", allow_pickle=True)

        # Get first sample
        offset = metadata[0]["offset"]
        length = metadata[0]["length"]
        sample_tokens = tokens[offset : offset + length].tolist()

        # Remove EOS
        sample_tokens = sample_tokens[:-1]

        # Find positions
        fp_pos = sample_tokens.index(fim_token_ids["prefix"])
        fs_pos = sample_tokens.index(fim_token_ids["suffix"])
        fm_pos = sample_tokens.index(fim_token_ids["middle"])

        # Verify order
        assert fp_pos < fs_pos < fm_pos


class TestFIMVsNonFIMSamples:
    """Test distinction between FIM and non-FIM samples."""

    def test_fim_vs_nonfim_distinction(self, temp_dir, test_tokenizer_with_fim, fim_token_ids):
        """Test that FIM and non-FIM samples can be distinguished."""
        config = UniversalDataConfig(
            datasets=[
                DatasetConfig(
                    name="test_mixed",
                    source="dummy",
                    task_type="pretrain",
                    fim_rate=0.5,  # 50% mix
                    fim_prefix_token="<fim_prefix>",
                    fim_suffix_token="<fim_suffix>",
                    fim_middle_token="<fim_middle>",
                    text_column="text",
                )
            ],
            vocab_name_or_path="gpt2",
            seq_length=512,
            preprocessed_dir=temp_dir / "preprocessed",
            cache_dir=temp_dir / "cache",
        )

        serializer = DataSerializer(config, test_tokenizer_with_fim, verbose=False)

        dataset = MockDataset([{"text": f"def function_{i}():\n    return {i}"} for i in range(30)])

        output_path = config.get_dataset_output_path(config.datasets[0])
        bin_path = output_path / "data.bin"
        idx_path = output_path / "data.idx"

        serializer._serialize_pretrain(dataset, config.datasets[0], bin_path, idx_path)

        # Load and categorize
        tokens = np.fromfile(bin_path, dtype=np.uint16)
        metadata = np.load(output_path / "data.idx.npy", allow_pickle=True)

        fim_count = 0
        non_fim_count = 0

        for meta in metadata:
            offset = meta["offset"]
            length = meta["length"]
            sample_tokens = tokens[offset : offset + length].tolist()

            has_fim = fim_token_ids["prefix"] in sample_tokens

            if has_fim:
                fim_count += 1
                # Verify PSM structure
                sample_no_eos = sample_tokens[:-1]
                assert verify_psm_structure(
                    sample_no_eos,
                    fim_token_ids["prefix"],
                    fim_token_ids["suffix"],
                    fim_token_ids["middle"],
                )
            else:
                non_fim_count += 1
                # Verify no FIM tokens
                assert fim_token_ids["prefix"] not in sample_tokens
                assert fim_token_ids["suffix"] not in sample_tokens
                assert fim_token_ids["middle"] not in sample_tokens

        # Should have both types
        assert fim_count > 0
        assert non_fim_count > 0

    def test_nonfim_metadata_type(self, temp_dir, test_tokenizer_with_fim):
        """
        KNOWN ISSUE: Test that metadata type is 'pretrain' for both FIM and non-FIM.

        Currently there's no distinction in metadata between FIM and non-FIM samples.
        Both have type='pretrain'. Future enhancement could add 'pretrain_fim' type.
        """
        config = UniversalDataConfig(
            datasets=[
                DatasetConfig(
                    name="test_metadata",
                    source="dummy",
                    task_type="pretrain",
                    fim_rate=0.5,
                    fim_prefix_token="<fim_prefix>",
                    fim_suffix_token="<fim_suffix>",
                    fim_middle_token="<fim_middle>",
                    text_column="text",
                )
            ],
            vocab_name_or_path="gpt2",
            seq_length=512,
            preprocessed_dir=temp_dir / "preprocessed",
            cache_dir=temp_dir / "cache",
        )

        serializer = DataSerializer(config, test_tokenizer_with_fim, verbose=False)

        dataset = MockDataset([{"text": f"def function_{i}():\n    return {i}"} for i in range(10)])

        output_path = config.get_dataset_output_path(config.datasets[0])
        bin_path = output_path / "data.bin"
        idx_path = output_path / "data.idx"

        serializer._serialize_pretrain(dataset, config.datasets[0], bin_path, idx_path)

        # Load metadata
        metadata = np.load(output_path / "data.idx.npy", allow_pickle=True)

        # All should have type='pretrain' (no distinction)
        for meta in metadata:
            assert meta["type"] == "pretrain"


class TestFIMMultipleDatasets:
    """Test FIM with multiple datasets."""

    def test_fim_multiple_datasets_different_rates(
        self, temp_dir, test_tokenizer_with_fim, fim_token_ids
    ):
        """Test multiple datasets with different FIM rates."""
        config = UniversalDataConfig(
            datasets=[
                DatasetConfig(
                    name="dataset_100",
                    source="dummy1",
                    task_type="pretrain",
                    fim_rate=1.0,  # 100%
                    fim_prefix_token="<fim_prefix>",
                    fim_suffix_token="<fim_suffix>",
                    fim_middle_token="<fim_middle>",
                    text_column="text",
                ),
                DatasetConfig(
                    name="dataset_0",
                    source="dummy2",
                    task_type="pretrain",
                    fim_rate=0.0,  # 0%
                    text_column="text",
                ),
            ],
            vocab_name_or_path="gpt2",
            seq_length=512,
            preprocessed_dir=temp_dir / "preprocessed",
            cache_dir=temp_dir / "cache",
        )

        serializer = DataSerializer(config, test_tokenizer_with_fim, verbose=False)

        # Serialize first dataset (100%)
        dataset1 = MockDataset(
            [{"text": f"Dataset 1 function {i}():\n    return {i}"} for i in range(10)]
        )

        output_path1 = config.get_dataset_output_path(config.datasets[0])
        bin_path1 = output_path1 / "data.bin"
        idx_path1 = output_path1 / "data.idx"

        serializer._serialize_pretrain(dataset1, config.datasets[0], bin_path1, idx_path1)

        # Serialize second dataset (0%)
        dataset2 = MockDataset(
            [{"text": f"Dataset 2 function {i}():\n    return {i}"} for i in range(10)]
        )

        output_path2 = config.get_dataset_output_path(config.datasets[1])
        bin_path2 = output_path2 / "data.bin"
        idx_path2 = output_path2 / "data.idx"

        serializer._serialize_pretrain(dataset2, config.datasets[1], bin_path2, idx_path2)

        # Verify first dataset has all FIM (np.save creates .npy extension)
        tokens1 = np.fromfile(bin_path1, dtype=np.uint16)
        metadata1 = np.load(output_path1 / "data.idx.npy", allow_pickle=True)

        fim_count1 = sum(
            1
            for meta in metadata1
            if fim_token_ids["prefix"] in tokens1[meta["offset"] : meta["offset"] + meta["length"]]
        )
        assert fim_count1 >= 9  # Most should be FIM (some might be too short)

        # Verify second dataset has no FIM (np.save creates .npy extension)
        tokens2 = np.fromfile(bin_path2, dtype=np.uint16)
        metadata2 = np.load(output_path2 / "data.idx.npy", allow_pickle=True)

        fim_count2 = sum(
            1
            for meta in metadata2
            if fim_token_ids["prefix"] in tokens2[meta["offset"] : meta["offset"] + meta["length"]]
        )
        assert fim_count2 == 0


class TestFIMErrorHandling:
    """Test error handling in FIM pipeline."""

    def test_fim_missing_tokens_error(self, temp_dir, test_tokenizer_without_fim):
        """Test error when FIM tokens missing from tokenizer."""
        config = UniversalDataConfig(
            datasets=[
                DatasetConfig(
                    name="test_error",
                    source="dummy",
                    task_type="pretrain",
                    fim_rate=0.5,
                    fim_prefix_token="<fim_prefix>",
                    fim_suffix_token="<fim_suffix>",
                    fim_middle_token="<fim_middle>",
                    text_column="text",
                )
            ],
            vocab_name_or_path="gpt2",
            seq_length=512,
            preprocessed_dir=temp_dir / "preprocessed",
            cache_dir=temp_dir / "cache",
        )

        serializer = DataSerializer(config, test_tokenizer_without_fim, verbose=False)

        dataset = MockDataset(
            [
                {"text": "def function():\n    x = 1\n    y = 2\n    z = 3\n    return x + y + z"},
            ]
        )

        output_path = config.get_dataset_output_path(config.datasets[0])
        bin_path = output_path / "data.bin"
        idx_path = output_path / "data.idx"

        # Should raise error during serialization
        with pytest.raises(ValueError) as exc_info:
            serializer._serialize_pretrain(dataset, config.datasets[0], bin_path, idx_path)

        assert "not found in tokenizer vocabulary" in str(exc_info.value)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
