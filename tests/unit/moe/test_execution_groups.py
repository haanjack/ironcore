# Copyright (c) 2025-2026 Jaegeun Han
# SPDX-License-Identifier: Apache-2.0

"""Row-budget coverage, expert boundaries, and idle-expert ownership."""

import pytest

from ironcore.config.config_moe import MoEConfig
from ironcore.layers.moe.execution_groups import ExecutionGroup, plan_execution_groups


@pytest.mark.parametrize("counts", [[0, 0, 0], [27, 0, 1, 4], [7, 7, 7, 7], [1, 100, 1], []])
@pytest.mark.parametrize("budget", [1, 7, 32])
def test_scheduler_covers_valid_rows_once(counts, budget):
    groups = plan_execution_groups(counts, budget)
    ownership = [expert for expert, count in enumerate(counts) for _ in range(count)]
    position = 0
    for group in groups:
        assert group.start == position
        assert 0 < group.tokens <= budget
        assert all(count > 0 for count in group.counts)
        assert len(set(group.experts)) == len(group.experts)
        rows = [
            expert
            for expert, count in zip(group.experts, group.counts, strict=True)
            for _ in range(count)
        ]
        assert rows == ownership[position : position + group.tokens]
        assert group.offsets[-1] == group.tokens
        position += group.tokens
    assert position == sum(counts)
    assert len(groups) == (sum(counts) + budget - 1) // budget
    assert all(group.tokens == budget for group in groups[:-1])


def test_group_boundary_splits_an_expert_and_fills_spare_rows():
    assert plan_execution_groups([5, 0, 3, 7], 7) == (
        ExecutionGroup(0, 7, (0, 2), (5, 2)),
        ExecutionGroup(7, 7, (2, 3), (1, 6)),
        ExecutionGroup(14, 1, (3,), (1,)),
    )


@pytest.mark.parametrize("counts", [[5, 0, 3, 7], [1, 100, 1], [7, 7, 7, 7], [0, 0]])
def test_preserve_expert_geometry_with_a_hard_row_budget(counts):
    budget = 7
    groups = plan_execution_groups(counts, budget, preserve_expert_segments=True)
    ownership = [expert for expert, count in enumerate(counts) for _ in range(count)]
    actual = []
    segments = {i: [] for i in range(len(counts))}
    for group in groups:
        assert group.start == len(actual)
        assert 0 < group.tokens <= budget
        for expert, count in zip(group.experts, group.counts, strict=True):
            actual.extend([expert] * count)
            segments[expert].append(count)
    assert actual == ownership
    for expert, count in enumerate(counts):
        if 0 < count <= budget:
            assert segments[expert] == [count]
    assert plan_execution_groups([5, 0, 3, 7], budget, preserve_expert_segments=True) == (
        ExecutionGroup(0, 5, (0,), (5,)),
        ExecutionGroup(5, 3, (2,), (3,)),
        ExecutionGroup(8, 7, (3,), (7,)),
    )


@pytest.mark.parametrize("budget", [0, -1, True, 2.5, "7", None])
def test_invalid_group_budget(budget):
    with pytest.raises(ValueError, match="positive integer"):
        MoEConfig(expert_backend="grouped", grouped_token_budget=budget)
    with pytest.raises(ValueError, match="positive integer"):
        plan_execution_groups([2], budget)


def test_single_row_group_budget_is_valid():
    config = MoEConfig(expert_backend="grouped", grouped_token_budget=1)
    assert len(plan_execution_groups([2, 1], config.grouped_token_budget)) == 3


@pytest.mark.parametrize("counts", [[2, -1], [True], [2.5]])
def test_invalid_counts(counts):
    with pytest.raises(ValueError, match="nonnegative"):
        plan_execution_groups(counts, 8)


def test_retired_virtual_block_option_is_rejected():
    with pytest.raises(KeyError, match="virtual_block_size"):
        MoEConfig(expert_backend="grouped")(virtual_block_size=128)


def test_grouped_requires_default_dispatch():
    from tests.fixtures.config_fixtures import create_moe_test_config

    from ironcore.layers.moe.moe_layer import CommunicationMode, MoEMLP

    config = create_moe_test_config()
    config.model.moe.expert_backend = "grouped"
    with pytest.raises(ValueError, match="default dispatch"):
        MoEMLP(config, communication_mode=CommunicationMode.ALL_TO_ALL)
