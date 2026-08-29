# Copyright (c) 2025-2026 Jaegeun Han
#
# SPDX-License-Identifier: Apache-2.0

"""
Unit tests for StreamingDataset._iter_sft_streaming's within-dataset shuffling.

Regression coverage: the SFT/DPO streaming path randomized which *dataset* a
sample was drawn from next (weighted mixing) but always read samples from
each individual dataset in strict sequential file order — shuffle_buffer_size
was computed but never referenced. The docstring claimed "shuffled indices
on-the-fly", which was false. Fixed via a per-dataset streaming shuffle
buffer.
"""

import numpy as np

from ironcore.dataloader.dataset import StreamingDataset


class _FakeBinaryDataset:
    """Minimal stand-in for StreamingBinaryDataset: samples are just their
    own sequential index, so we can later check the order they were yielded in."""

    def __init__(self, num_samples: int):
        self.num_samples = num_samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return {
            "token_ids": np.array([idx], dtype=np.int64),
            "metadata": {"type": "sft", "group_id": -1, "mask_ranges": []},
        }


def _make_streaming_dataset(num_samples: int, shuffle_buffer_size: int, seed: int = 1337):
    """Build a StreamingDataset for the sft path without touching disk, by
    bypassing __init__ and setting only the attributes _iter_sft_streaming needs."""
    ds = StreamingDataset.__new__(StreamingDataset)
    fake = _FakeBinaryDataset(num_samples)
    ds.datasets = [fake]
    ds.weights = [1.0]
    ds.mode = "sft"
    ds.seed = seed
    ds.split_ranges = {id(fake): (0, num_samples)}
    ds.shuffle_buffer_size = shuffle_buffer_size
    ds.rank = 0
    ds.world_size = 1
    return ds


class TestSftStreamingShuffle:
    def test_all_samples_yielded_exactly_once(self):
        """Shuffling must not drop or duplicate samples."""
        num_samples = 500
        ds = _make_streaming_dataset(num_samples, shuffle_buffer_size=64)

        yielded = [sample["token_ids"].item() for sample in ds._iter_sft_streaming()]

        assert len(yielded) == num_samples
        assert sorted(yielded) == list(range(num_samples))

    def test_within_dataset_order_is_not_strictly_sequential(self):
        """Regression: previously every sample from a single dataset came out
        in strict file order (0, 1, 2, ...) even though shuffle_buffer_size
        was computed. With shuffling enabled, the yielded order should differ
        from the original file order for a large enough sample count."""
        num_samples = 1000
        ds = _make_streaming_dataset(num_samples, shuffle_buffer_size=128)

        yielded = [sample["token_ids"].item() for sample in ds._iter_sft_streaming()]

        assert yielded != list(range(num_samples)), (
            "samples were yielded in strict file order — shuffle buffer is not being used"
        )

    def test_deterministic_given_same_seed(self):
        num_samples = 200
        ds_a = _make_streaming_dataset(num_samples, shuffle_buffer_size=32, seed=7)
        ds_b = _make_streaming_dataset(num_samples, shuffle_buffer_size=32, seed=7)

        order_a = [s["token_ids"].item() for s in ds_a._iter_sft_streaming()]
        order_b = [s["token_ids"].item() for s in ds_b._iter_sft_streaming()]

        assert order_a == order_b

    def test_small_dataset_smaller_than_buffer(self):
        """Buffer size larger than the dataset must not error and must still
        yield every sample exactly once."""
        num_samples = 10
        ds = _make_streaming_dataset(num_samples, shuffle_buffer_size=1000)

        yielded = [sample["token_ids"].item() for sample in ds._iter_sft_streaming()]

        assert sorted(yielded) == list(range(num_samples))

    def test_multi_dataset_mixing_still_yields_all_samples(self):
        ds = StreamingDataset.__new__(StreamingDataset)
        fake_a = _FakeBinaryDataset(50)
        fake_b = _FakeBinaryDataset(80)
        ds.datasets = [fake_a, fake_b]
        ds.weights = [0.5, 0.5]
        ds.mode = "sft"
        ds.seed = 42
        ds.split_ranges = {id(fake_a): (0, 50), id(fake_b): (0, 80)}
        ds.shuffle_buffer_size = 20
        ds.rank = 0
        ds.world_size = 1

        yielded = list(ds._iter_sft_streaming())
        assert len(yielded) == 130
