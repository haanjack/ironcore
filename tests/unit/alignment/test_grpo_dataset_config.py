# Copyright (c) 2025-2026 Jaegeun Han
#
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest

from ironcore.alignment import dataset as dataset_module


def _config(*, num_datasets: int = 1, max_positions: int = 128, max_new_tokens: int = 32):
    datasets = [
        SimpleNamespace(
            source=f"dataset-{index}.jsonl",
            prompt_column="question",
            answer_column="answer",
        )
        for index in range(num_datasets)
    ]
    return SimpleNamespace(
        data=SimpleNamespace(datasets=datasets, num_workers=0),
        model=SimpleNamespace(max_position_embeddings=max_positions),
        alignment=SimpleNamespace(
            generation=SimpleNamespace(max_new_tokens=max_new_tokens)
        ),
        trainer=SimpleNamespace(train_batch_size=4),
        init=SimpleNamespace(seed=17),
    )


def test_grpo_iterator_reserves_context_for_generation(monkeypatch):
    captured = {}

    def fake_dataloader(**kwargs):
        captured.update(kwargs)
        return [{"batch": True}]

    monkeypatch.setattr(dataset_module, "get_grpo_dataloader", fake_dataloader)

    iterator = dataset_module.get_grpo_data_iterator(
        _config(max_positions=128, max_new_tokens=32)
    )

    assert next(iterator) == {"batch": True}
    assert captured["max_prompt_length"] == 96


def test_grpo_iterator_rejects_multiple_datasets():
    with pytest.raises(ValueError, match="exactly one train dataset"):
        dataset_module.get_grpo_data_iterator(_config(num_datasets=2))


def test_grpo_iterator_rejects_generation_larger_than_context():
    with pytest.raises(ValueError, match="must be smaller"):
        dataset_module.get_grpo_data_iterator(
            _config(max_positions=32, max_new_tokens=32)
        )
