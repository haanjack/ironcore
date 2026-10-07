# Copyright (c) 2025-2026 Jaegeun Han
# SPDX-License-Identifier: Apache-2.0
"""Independent standard AdamW oracle, including epsilon-sensitive updates."""

import copy

import pytest
import torch

from ironcore.optimizer.adamw import AdamWOptimizer
from ironcore.optimizer.muon import MuonOptimizer


@pytest.mark.parametrize("kind", ["adamw", "muon-adamw", "cpu-offload"])
@pytest.mark.parametrize("amsgrad", [False, True])
@pytest.mark.parametrize("scale", [1.0, 1e-8, 1e-10])
def test_optimizer_matches_torch_adamw_and_resumes(kind, amsgrad, scale):
    torch.manual_seed(31)
    actual = torch.nn.Parameter(torch.randn(9))
    expected = torch.nn.Parameter(actual.detach().clone())
    kwargs = dict(lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.03, amsgrad=amsgrad)
    reference = torch.optim.AdamW([expected], **kwargs)
    if kind == "muon-adamw":
        optimizer = MuonOptimizer(
            [],
            [{"params": [actual], "amsgrad": amsgrad}],
            adamw_lr=kwargs["lr"],
            adamw_betas=kwargs["betas"],
            adamw_eps=kwargs["eps"],
            adamw_weight_decay=kwargs["weight_decay"],
        )
    else:
        optimizer = AdamWOptimizer(
            [actual], **kwargs, offload_enabled=kind == "cpu-offload", offload_min_param_elements=0
        )
    for step in range(7):
        grad = torch.randn_like(actual) * scale
        actual.grad = grad.clone()
        expected.grad = grad.clone()
        optimizer.step()
        reference.step()
        torch.testing.assert_close(actual, expected, rtol=2e-6, atol=2e-7)
        torch.testing.assert_close(
            optimizer.state[actual]["exp_avg"], reference.state[expected]["exp_avg"]
        )
        torch.testing.assert_close(
            optimizer.state[actual]["exp_avg_sq"], reference.state[expected]["exp_avg_sq"]
        )
        if step == 2:
            optimizer.load_state_dict(copy.deepcopy(optimizer.state_dict()))
            reference.load_state_dict(copy.deepcopy(reference.state_dict()))


@pytest.mark.parametrize("kind", ["adamw", "muon-adamw"])
def test_bfloat16_parameter_resume_preserves_original_fp32_moments(kind):
    parameter = torch.nn.Parameter(torch.zeros(9, dtype=torch.bfloat16))
    optimizer = (
        AdamWOptimizer([parameter])
        if kind == "adamw"
        else MuonOptimizer([], [{"params": [parameter]}])
    )
    parameter.grad = torch.full_like(parameter, 0.0123)
    optimizer.step()
    before = copy.deepcopy(optimizer.state_dict())
    optimizer.load_state_dict(copy.deepcopy(before))
    for key in ["exp_avg", "exp_avg_sq"]:
        assert optimizer.state[parameter][key].dtype == torch.float32
        torch.testing.assert_close(
            optimizer.state[parameter][key], before["state"][0][key], atol=0, rtol=0
        )
