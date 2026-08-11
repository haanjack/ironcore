# Copyright (c) 2025-2026 Jaegeun Han
#
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import torch

from ironcore.trainers import grpo_trainer as trainer_module
from ironcore.trainers.grpo_trainer import GRPOTrainer


def test_eval_step_generates_and_returns_mean_reward(monkeypatch):
    trainer = GRPOTrainer.__new__(GRPOTrainer)
    trainer.reward_worker = SimpleNamespace(
        score_batch=lambda **_: torch.tensor([1.0, 3.0])
    )
    trainer.use_chat_template = False
    trainer.system_prompt = None
    trainer.gen_kwargs = {"max_new_tokens": 4}
    trainer.model = object()
    trainer._tokenizer = SimpleNamespace(
        eos_token_id=2,
        batch_decode=lambda *_args, **_kwargs: ["first", "second"],
    )
    trainer._move_batch_to_device = lambda batch: batch
    trainer._get_compute_device = lambda: torch.device("cpu")

    captured = {}

    def fake_generate(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(
            response_ids=torch.tensor([[10], [11]]),
            metadata=[{"id": 1}, {"id": 2}],
        )

    monkeypatch.setattr(trainer_module, "generate_rollouts_batched", fake_generate)

    loss, reward = trainer._eval_step(
        {
            "input_ids": torch.tensor([[1], [2]]),
            "prompts": ["p1", "p2"],
            "metadata": [{"id": 1}, {"id": 2}],
        }
    )

    assert captured["group_size"] == 1
    assert loss == -2.0
    assert reward == 2.0
