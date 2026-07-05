# Copyright (c) 2025-2026 Jaegeun Han
#
# SPDX-License-Identifier: Apache-2.0

"""
Regression test for the TP=1 vs TP=2 weight-init divergence bug.

Root cause (fixed in ironcore/layers/module.py `_init_tp_weight`): every TP-sharded
weight used to be initialized from the SAME fixed seed (`config.init.seed`), so any
two equal-shaped TP weights (e.g. `linear_q` in different layers) received identical
values. Under TP>1 this meant every transformer layer started from the same degenerate
initialization.

The fix derives a per-parameter seed from `config.init.seed` + a hash of the
parameter name, using a local `torch.Generator` so the ambient RNG stream used for
non-TP params is left untouched.
"""

import torch

import ironcore.layers  # noqa: F401  (import before tensor_parallel to avoid a circular import)
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
from ironcore.layers.module import BaseModule
from ironcore.parallel.tensor_parallel import ColumnParallelLinear


def _make_config(seed: int = 1234) -> MainConfig:
    return MainConfig(
        model=ModelConfig(d_model=32, num_layers=2),
        init=InitConfig(seed=seed, init_std=0.02, xavier_init=False),
        optim=OptimConfig(),
        data=DataConfig(),
        parallel=ParallelConfig(),
        trainer=TrainerConfig(tensor_model_parallel_size=1),
        operation=OperationConfig(),
        utils=UtilsConfig(),
        profiler=ProfilerConfig(),
        peft=PEFTConfig(),
    )


class _TwoLayerModule(BaseModule):
    """Two same-shaped ColumnParallelLinear children, named like real TransformerLayers."""

    def __init__(self, config):
        super().__init__(config)
        self.layer0 = ColumnParallelLinear(config, input_size=32, output_size=32)
        self.layer1 = ColumnParallelLinear(config, input_size=32, output_size=32)


def test_same_shaped_tp_layers_get_different_weights():
    """Two equal-shaped TP weights must NOT be initialized identically."""
    config = _make_config()
    module = _TwoLayerModule(config)
    module.init_weights()

    assert not torch.allclose(module.layer0.weight, module.layer1.weight), (
        "layer0 and layer1 have identical shapes but received identical initial "
        "weights — the per-parameter seed derivation is broken (all TP layers "
        "would be degenerate under TP>1)."
    )


def test_tp_init_is_deterministic_given_same_seed():
    """Re-running init with the same seed must reproduce the same weights (for TP gather)."""
    config = _make_config(seed=777)

    module_a = _TwoLayerModule(config)
    module_a.init_weights()

    module_b = _TwoLayerModule(config)
    module_b.init_weights()

    assert torch.allclose(module_a.layer0.weight, module_b.layer0.weight)
    assert torch.allclose(module_a.layer1.weight, module_b.layer1.weight)


def test_tp_init_does_not_perturb_ambient_rng():
    """_init_tp_weight must not leak into the global torch RNG stream (regression:
    the old implementation called torch.manual_seed() globally and only restored
    state afterward, which worked but relied on careful save/restore; the local
    torch.Generator approach removes that risk entirely)."""
    config = _make_config()

    torch.manual_seed(999)
    before = torch.rand(4)

    torch.manual_seed(999)
    module = _TwoLayerModule(config)
    module.init_weights()
    after = torch.rand(4)

    assert torch.allclose(before, after), (
        "Ambient RNG stream was perturbed by TP weight initialization."
    )
