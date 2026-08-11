# Copyright (c) 2025-2026 Jaegeun Han
#
# SPDX-License-Identifier: Apache-2.0

import logging

from ironcore.alignment.rewards import RewardFunction, RewardManager
from ironcore.alignment.rewards import manager as manager_module
from ironcore.alignment.rewards.builtin import _stable_digest
from ironcore.config.config_alignment import (
    RewardFunctionEntry,
    RewardManagerConfig,
)


class _FailingReward(RewardFunction):
    def compute(self, prompt: str, completion: str, metadata: dict) -> float:
        raise RuntimeError("failed reward")


def test_api_reward_is_wired_from_config(monkeypatch):
    captured = {}

    class FakeAPIReward:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        def compute(self, prompt, completion, metadata):
            return 1.0

    monkeypatch.setattr(manager_module, "APIRewardFunction", FakeAPIReward)
    config = RewardManagerConfig(
        functions=[
            RewardFunctionEntry(
                name="judge", type="api", api_provider="anthropic", api_model="judge-model"
            )
        ]
    )

    manager = RewardManager.from_config(config)

    assert captured == {"provider": "anthropic", "model": "judge-model"}
    assert manager.compute("prompt", "completion", {}) == 1.0


def test_batch_failure_is_counted_and_logged(caplog):
    manager = RewardManager(num_workers=1, default_reward=0.25)
    manager.register("broken", _FailingReward())

    with caplog.at_level(logging.WARNING):
        rewards = manager.score_batch(["p"], ["c"], [{}])

    assert rewards.tolist() == [0.25]
    assert manager.failure_count == 1
    assert "failed reward" in caplog.text


def test_reward_cache_digest_is_stable_and_content_sensitive():
    assert _stable_digest("same") == _stable_digest("same")
    assert _stable_digest("same") != _stable_digest("different")
