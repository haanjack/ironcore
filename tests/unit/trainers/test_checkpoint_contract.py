# Copyright (c) 2025-2026 Jaegeun Han
# SPDX-License-Identifier: Apache-2.0
import json
import random
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from ironcore.checkpointing.integrity import digest_file, verify_manifest
from ironcore.parallel import parallel_states
from ironcore.trainers.checkpoint_state import load_trainer_state, save_trainer_state


class Cursor:
    def __init__(self):
        self.offset = 17

    def state_dict(self):
        return {"offset": self.offset}

    def load_state_dict(self, state):
        self.offset = state["offset"]


def test_trainer_checkpoint_restores_python_numpy_torch_and_data_cursor(tmp_path):
    parallel_states.initialize_model_parallel(1, 2)
    try:
        model = torch.nn.Linear(3, 7)
        trainer = SimpleNamespace(
            model=model,
            config=SimpleNamespace(
                trainer=SimpleNamespace(
                    model_path=str(tmp_path),
                    tensor_model_parallel_size=1,
                    parameter_precision="float32",
                ),
                operation=SimpleNamespace(no_save=False),
                data=SimpleNamespace(task_type="sft"),
                optim=SimpleNamespace(load_checkpoint_optim_state=True),
            ),
            data_iterator={"train": Cursor()},
            scaler=torch.amp.GradScaler("cpu", enabled=False),
        )
        save_trainer_state(trainer, 3)
        expected = (random.random(), float(np.random.rand()), torch.rand(8))
        trainer.data_iterator["train"].offset = 999
        for _ in range(10):
            random.random()
            np.random.rand()
            torch.rand(8)
        load_trainer_state(trainer, 3)
        actual = (random.random(), float(np.random.rand()), torch.rand(8))
        assert actual[:2] == expected[:2]
        torch.testing.assert_close(actual[2], expected[2], atol=0, rtol=0)
        assert trainer.data_iterator["train"].offset == 17
        trainer.config.data.task_type = "dpo"
        with pytest.raises(ValueError, match="original task"):
            load_trainer_state(trainer, 3)
    finally:
        parallel_states.destroy_model_parallel()


def test_manifest_detects_corruption_and_missing_sidecars(tmp_path):
    file = tmp_path / "model.pt"
    file.write_bytes(b"correct")
    (tmp_path / "manifest.json").write_text(json.dumps({"files": {"model.pt": digest_file(file)}}))
    verify_manifest(tmp_path, "manifest.json")
    file.write_bytes(b"corrupt")
    with pytest.raises(RuntimeError, match="integrity"):
        verify_manifest(tmp_path, "manifest.json")
    file.unlink()
    with pytest.raises(FileNotFoundError):
        verify_manifest(tmp_path, "manifest.json")


def test_failed_native_save_keeps_previous_committed_step(tmp_path, monkeypatch):
    import logging

    from tests.fixtures.config_fixtures import create_test_config

    from ironcore.checkpointing import native
    from ironcore.utils import Timer

    parallel_states.initialize_model_parallel(1, 2)
    try:
        config = create_test_config()
        config.trainer.model_path = str(tmp_path)
        monkeypatch.setattr(native, "get_logger", lambda: logging.getLogger(__name__))
        monkeypatch.setattr(native, "get_timer", Timer)
        model = torch.nn.Linear(3, 7)
        optimizer = torch.optim.AdamW(model.parameters())
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1)
        native.save_checkpoint(config, model, optimizer, scheduler, 1)
        assert (tmp_path / "latest_step.txt").read_text().strip() == "1"

        def fail(*args, **kwargs):
            raise OSError("Injected disk failure")

        monkeypatch.setattr(torch, "save", fail)
        with pytest.raises(OSError, match="disk failure"):
            native.save_checkpoint(config, model, optimizer, scheduler, 2)
        assert (tmp_path / "latest_step.txt").read_text().strip() == "1"
        assert not (tmp_path / "step_2" / "native_manifest.json").exists()
    finally:
        parallel_states.destroy_model_parallel()
