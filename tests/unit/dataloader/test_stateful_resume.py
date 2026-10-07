# Copyright (c) 2025-2026 Jaegeun Han
# SPDX-License-Identifier: Apache-2.0
import copy

import numpy as np
import pytest
import torch
from torchdata.stateful_dataloader import StatefulDataLoader

from ironcore.dataloader.collator import UniversalCollator
from ironcore.dataloader.dataset import StreamingBinaryDataset, StreamingDataset
from ironcore.dataloader.stateful import CheckpointableIterator


def dataset(path, mode="sft", rank=0, world=1):
    binary = path / "data.bin"
    index = path / "data.idx.npy"
    if not binary.exists():
        data = np.arange(400, dtype=np.uint16).reshape(100, 4)
        data.tofile(binary)
        meta = np.zeros(
            100,
            dtype=[
                ("offset", "i8"),
                ("length", "i8"),
                ("type", "U8"),
                ("group_id", "i8"),
                ("mask_ranges", "U32"),
            ],
        )
        meta["offset"] = np.arange(100) * 4
        meta["length"] = 4
        meta["type"] = "sft"
        meta["mask_ranges"] = "[]"
        np.save(index, meta)
    source = StreamingBinaryDataset(binary, index)
    ds = StreamingDataset.__new__(StreamingDataset)
    ds.datasets = [source]
    ds.weights = [1.0]
    ds.mode = mode
    ds.split = "train"
    ds.seed = 81
    ds.seq_length = 4
    ds.shuffle_buffer_size = 13
    ds.rank = rank
    ds.world_size = world
    ds.split_ranges = {id(source): (0, source.total_tokens if mode == "pretrain" else len(source))}
    return ds


@pytest.mark.parametrize("mode", ["pretrain", "sft"])
def test_direct_cursor_seeks_without_replaying_samples(tmp_path, mode):
    ds = dataset(tmp_path, mode)
    stream = iter(ds)
    for _ in range(17):
        next(stream)
    state = copy.deepcopy(ds.state_dict())
    torch.save(state, tmp_path / "cursor.pt")
    restored = dataset(tmp_path, mode)
    restored.load_state_dict(torch.load(tmp_path / "cursor.pt", weights_only=True))
    resumed = iter(restored)
    for _ in range(25):
        a, b = next(stream), next(resumed)
        torch.testing.assert_close(
            a if mode == "pretrain" else a["token_ids"],
            b if mode == "pretrain" else b["token_ids"],
            atol=0,
            rtol=0,
        )


def test_sft_dp_ranks_partition_one_global_permutation(tmp_path):
    expected = [int(s["token_ids"][0]) for s in dataset(tmp_path)]
    ranks = [[int(s["token_ids"][0]) for s in dataset(tmp_path, rank=r, world=2)] for r in [0, 1]]
    assert ranks[0] == expected[::2]
    assert ranks[1] == expected[1::2]
    assert set(ranks[0]).isdisjoint(ranks[1])


@pytest.mark.parametrize("workers", [0, 2])
def test_consumed_batch_checkpoint_includes_worker_prefetch_and_packing(tmp_path, workers):
    def loader():
        return CheckpointableIterator(
            StatefulDataLoader(
                dataset(tmp_path),
                batch_size=5,
                num_workers=workers,
                multiprocessing_context="spawn" if workers else None,
                snapshot_every_n_steps=1,
                collate_fn=UniversalCollator("sft", 16, return_full_attention_mask=True),
            )
        )

    a = loader()
    for _ in range(3):
        next(a)
    state = copy.deepcopy(a.state_dict())
    torch.save(state, tmp_path / "loader.pt")
    b = loader()
    b.load_state_dict(torch.load(tmp_path / "loader.pt", weights_only=True))
    for _ in range(8):
        actual, expected = next(a), next(b)
        for key in ["input_ids", "labels", "loss_sample_ids", "position_ids", "attention_mask"]:
            torch.testing.assert_close(actual[key], expected[key], atol=0, rtol=0)
    if workers:
        a.iterator._shutdown_workers()
        b.iterator._shutdown_workers()


def test_changed_data_topology_is_rejected(tmp_path):
    ds = dataset(tmp_path)
    next(iter(ds))
    restored = dataset(tmp_path, rank=1, world=2)
    with pytest.raises(ValueError, match="same files"):
        restored.load_state_dict(ds.state_dict())


@pytest.mark.parametrize("workers", [0, 2])
def test_grpo_loader_resumes_shuffled_epochs_and_prefetched_prompts(tmp_path, workers):
    import json

    from ironcore.alignment.dataset import GRPODataset, collate_grpo_samples

    class Tokenizer:
        def __call__(self, text, **kwargs):
            ids = torch.tensor([[int(text), 2]])
            return {"input_ids": ids, "attention_mask": torch.ones_like(ids)}

    path = tmp_path / "prompts.json"
    path.write_text(json.dumps([{"prompt": str(i), "answer": str(i)} for i in range(31)]))

    def loader():
        return CheckpointableIterator(
            StatefulDataLoader(
                GRPODataset(path, tokenizer=Tokenizer()),
                batch_size=3,
                num_workers=workers,
                collate_fn=collate_grpo_samples,
                snapshot_every_n_steps=1,
            )
        )

    a = loader()
    for _ in range(15):
        next(a)
    b = loader()
    b.load_state_dict(copy.deepcopy(a.state_dict()))
    for _ in range(20):
        actual, expected = next(a), next(b)
        assert actual["prompts"] == expected["prompts"]
        torch.testing.assert_close(actual["input_ids"], expected["input_ids"], atol=0, rtol=0)
    if workers:
        a.iterator._shutdown_workers()
        b.iterator._shutdown_workers()
