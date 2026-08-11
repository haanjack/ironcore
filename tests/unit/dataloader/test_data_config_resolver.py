# Copyright (c) 2025-2026 Jaegeun Han
#
# SPDX-License-Identifier: Apache-2.0

import logging
from types import SimpleNamespace

import pytest

from ironcore.dataloader.data_config import resolve_data_config


def test_resolver_loads_data_config_anywhere_under_configs():
    config = SimpleNamespace(
        data=SimpleNamespace(config_path="configs/experiments/nanogpt/data.yaml"),
        model=SimpleNamespace(max_seq_len=1024),
    )

    resolved = resolve_data_config(config)

    assert resolved.seq_length == 1024
    assert resolved.datasets[0].name == "openwebtext"


def test_resolver_rejects_paths_outside_configs(tmp_path):
    config = SimpleNamespace(
        data=SimpleNamespace(config_path=str(tmp_path / "data.yaml")),
        model=SimpleNamespace(max_seq_len=128),
    )

    with pytest.raises(ValueError, match="outside allowed directory"):
        resolve_data_config(config)


def test_resolver_warns_on_sequence_length_mismatch(caplog):
    data = SimpleNamespace(seq_length=64, datasets=[])
    config = SimpleNamespace(data=data, model=SimpleNamespace(max_seq_len=128))

    with caplog.at_level(logging.WARNING):
        assert resolve_data_config(config) is data

    assert "differs from model max_seq_len" in caplog.text
