# Copyright (c) 2025-2026 Jaegeun Han
#
# SPDX-License-Identifier: Apache-2.0

"""
Unit tests for ironcore/optimizer/lr_scheduler.py.

Regression coverage for the previously-broken `linear` scheduler path
(get_lr_scheduler returned the LinearDecayLRScheduler *class* instead of an
instance, with ctor kwargs that didn't even match its signature and a
last_epoch=total_steps bug), plus the warmup_steps=0 default (division by
zero when the warmup branch was taken at step 0).
"""

import logging
from types import SimpleNamespace

import pytest
import torch

from ironcore.optimizer.lr_scheduler import (
    CosineAnnealingLR,
    LinearDecayLRScheduler,
    get_lr_scheduler,
)


@pytest.fixture(autouse=True)
def _stub_logger(monkeypatch):
    """get_lr_scheduler() calls get_logger(), which asserts on a process-wide
    GlobalStates singleton normally set up by the full training entrypoint.
    Stub it out so these tests can exercise get_lr_scheduler() in isolation."""
    import ironcore.optimizer.lr_scheduler as lr_scheduler_module

    monkeypatch.setattr(lr_scheduler_module, "get_logger", lambda: logging.getLogger("test"))


def _make_config():
    return SimpleNamespace(
        optim=SimpleNamespace(
            lr_scheduler="cosine",
            max_lr=1e-3,
            min_lr=1e-5,
            warmup_steps=0,
            annealing_steps=0,
        ),
        operation=SimpleNamespace(train_steps=100),
    )


def _make_optimizer(lr=1e-3):
    param = torch.nn.Parameter(torch.zeros(4))
    return torch.optim.SGD([param], lr=lr)


class TestGetLrScheduler:
    def test_linear_returns_an_instance_not_the_class(self):
        cfg = _make_config()
        cfg.optim.lr_scheduler = "linear"
        optimizer = _make_optimizer()

        scheduler = get_lr_scheduler(cfg, optimizer)

        assert isinstance(scheduler, LinearDecayLRScheduler)
        # The old bug returned the class object itself.
        assert not isinstance(scheduler, type)

    def test_linear_scheduler_step_does_not_raise(self):
        cfg = _make_config()
        cfg.optim.lr_scheduler = "linear"
        cfg.optim.warmup_steps = 10
        optimizer = _make_optimizer()

        scheduler = get_lr_scheduler(cfg, optimizer)
        for _ in range(20):
            optimizer.step()
            scheduler.step()

        assert all(torch.isfinite(torch.tensor(lr)) for lr in scheduler.get_last_lr())

    def test_cosine_returns_an_instance(self):
        cfg = _make_config()
        cfg.optim.lr_scheduler = "cosine"
        optimizer = _make_optimizer()

        scheduler = get_lr_scheduler(cfg, optimizer)

        assert isinstance(scheduler, CosineAnnealingLR)

    def test_unknown_scheduler_raises(self):
        cfg = _make_config()
        cfg.optim.lr_scheduler = "does-not-exist"
        optimizer = _make_optimizer()

        with pytest.raises(NotImplementedError):
            get_lr_scheduler(cfg, optimizer)


class TestLinearDecayLRScheduler:
    def test_warmup_then_decay_to_zero(self):
        optimizer = _make_optimizer(lr=1.0)
        scheduler = LinearDecayLRScheduler(optimizer, warmup_steps=5, total_steps=10)

        lrs = []
        for _ in range(10):
            optimizer.step()
            scheduler.step()
            lrs.append(scheduler.get_last_lr()[0])

        # Warms up to the base LR, then decays toward 0 by total_steps.
        assert lrs[4] > lrs[0]
        assert lrs[-1] < lrs[4]
        assert all(lr >= -1e-9 for lr in lrs)

    def test_zero_warmup_steps_does_not_raise(self):
        """warmup_steps=0 is the OptimConfig default; must not divide by zero."""
        optimizer = _make_optimizer(lr=1.0)
        scheduler = LinearDecayLRScheduler(optimizer, warmup_steps=0, total_steps=10)

        for _ in range(10):
            optimizer.step()
            scheduler.step()

        assert torch.isfinite(torch.tensor(scheduler.get_last_lr()[0]))

    def test_last_epoch_starts_fresh_not_at_total_steps(self):
        """Regression: the old code passed last_epoch=total_steps to super().__init__,
        which should have been -1 (a fresh schedule)."""
        optimizer = _make_optimizer(lr=1.0)
        scheduler = LinearDecayLRScheduler(optimizer, warmup_steps=0, total_steps=1000)

        assert scheduler.last_epoch < 10


class TestCosineAnnealingLR:
    def test_zero_warmup_steps_does_not_raise(self):
        optimizer = _make_optimizer(lr=1.0)
        scheduler = CosineAnnealingLR(
            optimizer, warmup_steps=0, annealing_steps=10, total_steps=10, max_lr=1.0, min_lr=0.1
        )

        for _ in range(10):
            optimizer.step()
            scheduler.step()

        assert torch.isfinite(torch.tensor(scheduler.get_last_lr()[0]))

    def test_decays_toward_min_lr(self):
        optimizer = _make_optimizer(lr=1.0)
        scheduler = CosineAnnealingLR(
            optimizer, warmup_steps=0, annealing_steps=20, total_steps=20, max_lr=1.0, min_lr=0.1
        )

        lrs = []
        for _ in range(20):
            optimizer.step()
            scheduler.step()
            lrs.append(scheduler.get_last_lr()[0])

        assert lrs[0] > lrs[-1]
        assert lrs[-1] == pytest.approx(0.1, abs=1e-6)

    def test_max_lr_less_than_min_lr_raises(self):
        optimizer = _make_optimizer(lr=1.0)
        with pytest.raises(ValueError):
            CosineAnnealingLR(
                optimizer, warmup_steps=0, annealing_steps=10, total_steps=10, max_lr=0.1, min_lr=1.0
            )
