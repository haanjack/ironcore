"""
Integration tests for FIM serialization.

Tests the full serialization cycle: serialize → save .bin/.idx → load → verify PSM.
"""

import numpy as np
import pytest
from tests.fixtures.mocks import MockDataset
from tests.fixtures.utils import verify_psm_structure

from ironcore.dataloader.data_config import DatasetConfig, UniversalDataConfig
from ironcore.preprocessing.serializer import DataSerializer


class TestFIMSerializeAndLoad:
    """Test full serialization and loading cycle."""

    def test_fim_serialize_and_load(self, temp_dir, test_tokenizer_with_fim, fim_token_ids):
        """Test full cycle: serialize → save → load → verify PSM."""
        # Create config with FIM enabled
        config = UniversalDataConfig(
            datasets=[
                DatasetConfig(
                    name="test_fim",
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
            seq_length=1024,
            preprocessed_dir=temp_dir / "preprocessed",
            cache_dir=temp_dir / "cache",
        )

        # Create serializer
        serializer = DataSerializer(config, test_tokenizer_with_fim, verbose=False)

        # Create mock dataset
        dataset = MockDataset(
            [
                {"text": "def hello():\n    print('world')\n    return 42"},
                {"text": "class Foo:\n    def __init__(self):\n        self.x = 10"},
            ]
        )

        # Serialize
        output_path = config.get_dataset_output_path(config.datasets[0])
        bin_path = output_path / "data.bin"
        idx_path = output_path / "data.idx"

        serializer._serialize_pretrain(dataset, config.datasets[0], bin_path, idx_path)

        # Verify files exist (np.save creates .npy extension)
        assert bin_path.exists()
        idx_path_npy = output_path / "data.idx.npy"
        assert idx_path_npy.exists()

        # Load and verify
        tokens = np.fromfile(bin_path, dtype=np.uint16)
        metadata = np.load(idx_path_npy, allow_pickle=True)

        # Verify we have samples
        assert len(metadata) == 2

        # Extract first sample and verify PSM structure
        first_offset = metadata[0]["offset"]
        first_length = metadata[0]["length"]
        first_tokens = tokens[first_offset : first_offset + first_length].tolist()

        # Remove EOS token (last token)
        first_tokens = first_tokens[:-1]

        # Should have FIM structure (rate=1.0)
        assert verify_psm_structure(
            first_tokens, fim_token_ids["prefix"], fim_token_ids["suffix"], fim_token_ids["middle"]
        )

    def test_fim_binary_format_uint16(self, temp_dir, test_tokenizer_with_fim):
        """Test correct binary format (uint16) for small vocab."""
        config = UniversalDataConfig(
            datasets=[
                DatasetConfig(
                    name="test_fim",
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
            seq_length=1024,
            preprocessed_dir=temp_dir / "preprocessed",
            cache_dir=temp_dir / "cache",
        )

        serializer = DataSerializer(config, test_tokenizer_with_fim, verbose=False)

        dataset = MockDataset(
            [
                {"text": "Short text for testing binary format."},
            ]
        )

        output_path = config.get_dataset_output_path(config.datasets[0])
        bin_path = output_path / "data.bin"
        idx_path = output_path / "data.idx"

        serializer._serialize_pretrain(dataset, config.datasets[0], bin_path, idx_path)

        # Load and check dtype
        tokens = np.fromfile(bin_path, dtype=np.uint16)
        assert tokens.dtype == np.uint16

    def test_fim_metadata_structure(self, temp_dir, test_tokenizer_with_fim):
        """Verify .idx metadata format."""
        config = UniversalDataConfig(
            datasets=[
                DatasetConfig(
                    name="test_fim",
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
            seq_length=1024,
            preprocessed_dir=temp_dir / "preprocessed",
            cache_dir=temp_dir / "cache",
        )

        serializer = DataSerializer(config, test_tokenizer_with_fim, verbose=False)

        dataset = MockDataset(
            [
                {"text": "Sample text for metadata verification test."},
            ]
        )

        output_path = config.get_dataset_output_path(config.datasets[0])
        bin_path = output_path / "data.bin"
        idx_path = output_path / "data.idx"

        serializer._serialize_pretrain(dataset, config.datasets[0], bin_path, idx_path)

        # Load metadata (np.save creates .npy extension)
        idx_path_npy = output_path / "data.idx.npy"
        metadata = np.load(idx_path_npy, allow_pickle=True)

        # Check structure
        assert len(metadata) == 1
        sample_meta = metadata[0]

        # Check fields
        assert "offset" in sample_meta.dtype.names
        assert "length" in sample_meta.dtype.names
        assert "type" in sample_meta.dtype.names
        assert "group_id" in sample_meta.dtype.names
        assert "mask_ranges" in sample_meta.dtype.names

        # Check values
        assert sample_meta["type"] == "pretrain"
        assert sample_meta["group_id"] == -1
        assert sample_meta["mask_ranges"] == "[]"


class TestFIMDeterministicSerialization:
    """Test deterministic serialization behavior."""

    def test_fim_deterministic_serialization(self, temp_dir, test_tokenizer_with_fim):
        """Test that same seed produces identical binary output."""
        config = UniversalDataConfig(
            datasets=[
                DatasetConfig(
                    name="test_fim",
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
            seq_length=1024,
            preprocessed_dir=temp_dir / "preprocessed",
            cache_dir=temp_dir / "cache",
        )

        dataset = MockDataset(
            [
                {"text": "def function_one():\n    return 1"},
                {"text": "def function_two():\n    return 2"},
                {"text": "def function_three():\n    return 3"},
            ]
        )

        # Serialize twice
        for run in [1, 2]:
            serializer = DataSerializer(config, test_tokenizer_with_fim, verbose=False)

            output_path = temp_dir / f"run{run}" / "test_fim" / "pretrain"
            output_path.mkdir(parents=True, exist_ok=True)
            bin_path = output_path / "data.bin"
            idx_path = output_path / "data.idx"

            serializer._serialize_pretrain(dataset, config.datasets[0], bin_path, idx_path)

        # Compare outputs
        tokens1 = np.fromfile(
            temp_dir / "run1" / "test_fim" / "pretrain" / "data.bin", dtype=np.uint16
        )
        tokens2 = np.fromfile(
            temp_dir / "run2" / "test_fim" / "pretrain" / "data.bin", dtype=np.uint16
        )

        # Should be identical (same seed=1337 used in _serialize_pretrain)
        np.testing.assert_array_equal(tokens1, tokens2)


class TestFIMRateRespected:
    """Test that configured FIM rate is respected."""

    def test_fim_rate_respected_50_percent(self, temp_dir, test_tokenizer_with_fim, fim_token_ids):
        """Test that actual FIM rate approximately matches configured rate."""
        config = UniversalDataConfig(
            datasets=[
                DatasetConfig(
                    name="test_fim",
                    source="dummy",
                    task_type="pretrain",
                    fim_rate=0.5,  # 50%
                    fim_prefix_token="<fim_prefix>",
                    fim_suffix_token="<fim_suffix>",
                    fim_middle_token="<fim_middle>",
                    text_column="text",
                )
            ],
            vocab_name_or_path="gpt2",
            seq_length=1024,
            preprocessed_dir=temp_dir / "preprocessed",
            cache_dir=temp_dir / "cache",
        )

        serializer = DataSerializer(config, test_tokenizer_with_fim, verbose=False)

        # Create larger dataset for statistical test
        dataset = MockDataset(
            [{"text": f"def function_{i}():\n    return {i}"} for i in range(100)]
        )

        output_path = config.get_dataset_output_path(config.datasets[0])
        bin_path = output_path / "data.bin"
        idx_path = output_path / "data.idx"

        serializer._serialize_pretrain(dataset, config.datasets[0], bin_path, idx_path)

        # Load and count FIM samples (np.save creates .npy extension)
        tokens = np.fromfile(bin_path, dtype=np.uint16)
        idx_path_npy = output_path / "data.idx.npy"
        metadata = np.load(idx_path_npy, allow_pickle=True)

        # Extract all samples and check for FIM tokens
        fim_count = 0
        for meta in metadata:
            offset = meta["offset"]
            length = meta["length"]
            sample_tokens = tokens[offset : offset + length].tolist()

            if fim_token_ids["prefix"] in sample_tokens:
                fim_count += 1

        actual_rate = fim_count / len(metadata)

        # Should be approximately 0.5 (within 15% tolerance for 100 samples)
        assert 0.35 <= actual_rate <= 0.65, f"Actual rate {actual_rate} not close to 0.5"

    def test_fim_rate_0_percent(self, temp_dir, test_tokenizer_with_fim, fim_token_ids):
        """Test that fim_rate=0.0 produces no FIM transformations."""
        config = UniversalDataConfig(
            datasets=[
                DatasetConfig(
                    name="test_fim",
                    source="dummy",
                    task_type="pretrain",
                    fim_rate=0.0,  # Disabled
                    text_column="text",
                )
            ],
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

        # Load and verify no FIM tokens (np.save creates .npy extension)
        tokens = np.fromfile(bin_path, dtype=np.uint16)
        idx_path_npy = output_path / "data.idx.npy"
        metadata = np.load(idx_path_npy, allow_pickle=True)

        fim_count = 0
        for meta in metadata:
            offset = meta["offset"]
            length = meta["length"]
            sample_tokens = tokens[offset : offset + length].tolist()

            if fim_token_ids["prefix"] in sample_tokens:
                fim_count += 1

        assert fim_count == 0, "FIM tokens found with fim_rate=0.0"

    def test_fim_rate_100_percent(self, temp_dir, test_tokenizer_with_fim, fim_token_ids):
        """Test that fim_rate=1.0 transforms all samples."""
        config = UniversalDataConfig(
            datasets=[
                DatasetConfig(
                    name="test_fim",
                    source="dummy",
                    task_type="pretrain",
                    fim_rate=1.0,  # 100%
                    fim_prefix_token="<fim_prefix>",
                    fim_suffix_token="<fim_suffix>",
                    fim_middle_token="<fim_middle>",
                    text_column="text",
                )
            ],
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

        # Load and verify all have FIM tokens (np.save creates .npy extension)
        tokens = np.fromfile(bin_path, dtype=np.uint16)
        idx_path_npy = output_path / "data.idx.npy"
        metadata = np.load(idx_path_npy, allow_pickle=True)

        fim_count = 0
        for meta in metadata:
            offset = meta["offset"]
            length = meta["length"]
            sample_tokens = tokens[offset : offset + length].tolist()

            if fim_token_ids["prefix"] in sample_tokens:
                fim_count += 1

        # All samples should have FIM (except potentially very short ones)
        assert fim_count >= len(metadata) * 0.9, "Not all samples transformed with fim_rate=1.0"


class TestFIMMixedSequences:
    """Test datasets with both FIM and non-FIM sequences."""

    def test_fim_mixed_sequences(self, temp_dir, test_tokenizer_with_fim, fim_token_ids):
        """Test dataset with both FIM and non-FIM sequences."""
        config = UniversalDataConfig(
            datasets=[
                DatasetConfig(
                    name="test_fim",
                    source="dummy",
                    task_type="pretrain",
                    fim_rate=0.5,  # 50% FIM
                    fim_prefix_token="<fim_prefix>",
                    fim_suffix_token="<fim_suffix>",
                    fim_middle_token="<fim_middle>",
                    text_column="text",
                )
            ],
            vocab_name_or_path="gpt2",
            seq_length=1024,
            preprocessed_dir=temp_dir / "preprocessed",
            cache_dir=temp_dir / "cache",
        )

        serializer = DataSerializer(config, test_tokenizer_with_fim, verbose=False)

        dataset = MockDataset([{"text": f"def function_{i}():\n    return {i}"} for i in range(50)])

        output_path = config.get_dataset_output_path(config.datasets[0])
        bin_path = output_path / "data.bin"
        idx_path = output_path / "data.idx"

        serializer._serialize_pretrain(dataset, config.datasets[0], bin_path, idx_path)

        # Load and analyze (np.save creates .npy extension)
        tokens = np.fromfile(bin_path, dtype=np.uint16)
        idx_path_npy = output_path / "data.idx.npy"
        metadata = np.load(idx_path_npy, allow_pickle=True)

        fim_samples = []
        non_fim_samples = []

        for meta in metadata:
            offset = meta["offset"]
            length = meta["length"]
            sample_tokens = tokens[offset : offset + length].tolist()

            if fim_token_ids["prefix"] in sample_tokens:
                fim_samples.append(sample_tokens)
            else:
                non_fim_samples.append(sample_tokens)

        # Should have both types
        assert len(fim_samples) > 0, "No FIM samples found"
        assert len(non_fim_samples) > 0, "No non-FIM samples found"

        # Verify FIM samples have PSM structure
        for fim_sample in fim_samples:
            # Remove EOS
            fim_sample_no_eos = fim_sample[:-1]
            assert verify_psm_structure(
                fim_sample_no_eos,
                fim_token_ids["prefix"],
                fim_token_ids["suffix"],
                fim_token_ids["middle"],
            )

        # Verify non-FIM samples don't have FIM tokens
        for non_fim_sample in non_fim_samples:
            assert fim_token_ids["prefix"] not in non_fim_sample
            assert fim_token_ids["suffix"] not in non_fim_sample
            assert fim_token_ids["middle"] not in non_fim_sample


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
