# Copyright (c) 2025-2026 Jaegeun Han
#
# SPDX-License-Identifier: Apache-2.0

"""
Regression test: universal-checkpoint save must not crash when the model has
frozen parameters (e.g. LoRA base_layer weights).

Root cause (fixed in ironcore/checkpointing/native.py save_checkpoint):
frozen parameters are never registered with the optimizer (get_optimizer /
get_muon_optimizer both skip `requires_grad=False` params), so
optimizer.state_dict()["state"] has fewer entries than
save_model.named_parameters(). The universal-checkpoint path zipped these
two sequences with strict=True, which raises as soon as the model has any
frozen parameter — which, with LoRA, is most of the model. Fixed by
filtering to trainable parameters before zipping.
"""

import logging

import torch

from ironcore.checkpointing.native import save_checkpoint
from ironcore.config import (
    DataConfig,
    InitConfig,
    MainConfig,
    ModelConfig,
    OperationConfig,
    OptimConfig,
    ParallelConfig,
    PEFTConfig,
    ProfilerConfig,
    TrainerConfig,
    UtilsConfig,
)
from ironcore.parallel import parallel_states


class _TinyModelWithFrozenParam(torch.nn.Module):
    """Stand-in for a LoRA-wrapped model: one trainable Linear plus a frozen
    parameter that is never registered with the optimizer."""

    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(4, 4)
        self.frozen_base_weight = torch.nn.Parameter(torch.zeros(4, 4))
        self.frozen_base_weight.requires_grad_(False)


def _make_config(tmp_path) -> MainConfig:
    return MainConfig(
        model=ModelConfig(d_model=4, num_layers=1),
        init=InitConfig(),
        optim=OptimConfig(),
        data=DataConfig(),
        parallel=ParallelConfig(),
        trainer=TrainerConfig(model_path=str(tmp_path)),
        operation=OperationConfig(save_dist_ckpt=False),
        utils=UtilsConfig(),
        profiler=ProfilerConfig(),
        peft=PEFTConfig(),
    )


class _FakeLRScheduler:
    def state_dict(self):
        return {}


def test_universal_checkpoint_save_with_frozen_params_does_not_raise(tmp_path, monkeypatch):
    model = _TinyModelWithFrozenParam()

    # Optimizer only ever sees trainable params, matching get_optimizer's
    # `if not p.requires_grad: continue` filtering.
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=1e-3)

    # Populate optimizer.state by taking one real step.
    for p in trainable_params:
        p.grad = torch.ones_like(p)
    optimizer.step()
    optimizer.zero_grad()

    # Force the universal-checkpoint code path (not config.operation.save_dist_ckpt,
    # and TP world size > 1) without needing a real distributed process group.
    monkeypatch.setattr(parallel_states, "get_tensor_model_parallel_world_size", lambda: 2)
    monkeypatch.setattr(parallel_states, "get_tensor_model_parallel_rank", lambda: 0)
    monkeypatch.setattr(parallel_states, "get_data_parallel_group_rank", lambda: 0)
    monkeypatch.setattr(parallel_states, "get_tensor_model_parallel_group", lambda: None)

    import ironcore.checkpointing.native as native_module
    from ironcore.utils.timer import Timer

    monkeypatch.setattr(native_module.dist, "barrier", lambda *args, **kwargs: None)
    monkeypatch.setattr(native_module, "get_logger", lambda: logging.getLogger("test"))
    monkeypatch.setattr(native_module, "get_timer", Timer)

    config = _make_config(tmp_path)

    # Must not raise (previously: ValueError from zip(..., strict=True) length mismatch,
    # and separately a KeyError from a frozen param corrupting the live optimizer's state).
    save_checkpoint(config, model, optimizer, _FakeLRScheduler(), step=1)

    assert len(list(tmp_path.rglob("step_1"))) == 1, "checkpoint step_1 directory was not created"
