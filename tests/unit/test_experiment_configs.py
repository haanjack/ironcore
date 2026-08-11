# Copyright (c) 2025-2026 Jaegeun Han
#
# SPDX-License-Identifier: Apache-2.0

from ironcore.train import load_full_config


def test_nanogpt_convergence_config_loads():
    config = load_full_config("configs/experiments/nanogpt/convergence.yaml")
    assert config.model.d_model == 768
    assert config.data.seq_length == 1024
    assert config.data.datasets[0].name == "openwebtext"


def test_nanogpt_dp2_config_loads(monkeypatch):
    monkeypatch.setenv("WORLD_SIZE", "2")
    config = load_full_config("configs/experiments/nanogpt_convergence_dp2.yaml")
    assert config.trainer.micro_batch_size == 12
    assert config.trainer.gradient_accumulation_steps == 20
    assert config.model.d_model == 768
