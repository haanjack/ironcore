# Copyright (c) 2025-2026 Jaegeun Han
# SPDX-License-Identifier: Apache-2.0

"""Virtual ownership and grouped execution coverage, including skew and idle experts."""

import pytest

from ironcore.config.config_moe import MoEConfig
from ironcore.layers.moe.virtual_blocks import plan_virtual_blocks


@pytest.mark.parametrize("counts", [[0, 0, 0], [27, 0, 1, 4], [7, 7, 7, 7], [1, 100, 1], []])
@pytest.mark.parametrize("block,budget", [(1, 1), (3, 7), (8, 32)])
def test_scheduler_covers_valid_rows_once(counts, block, budget):
    plan = plan_virtual_blocks(counts, block, budget)
    ownership = [expert for expert, count in enumerate(counts) for _ in range(count)]
    tiles = [tile.expert for tile in plan.blocks for _ in range(tile.tokens)]
    assert tiles == ownership
    position = 0
    for group in plan.groups:
        assert group.start == position
        assert 0 < group.tokens <= budget
        rows = [
            expert
            for expert, count in zip(group.experts, group.counts, strict=True)
            for _ in range(count)
        ]
        assert rows == ownership[position : position + group.tokens]
        assert group.offsets[-1] == group.tokens
        position += group.tokens
    assert position == sum(counts)


def test_logical_tiles_coalesce_for_larger_gemms():
    plan = plan_virtual_blocks([16, 8], 2, 12)
    assert len(plan.blocks) == 12
    assert len(plan.groups) == 2
    assert plan.groups[0].counts == (12,)
    assert plan.groups[1].counts == (4, 8)


@pytest.mark.parametrize("block,budget", [(0, 4), (True, 4), (2, 1), (2, True), (2, 2.5)])
def test_invalid_scheduler_config(block, budget):
    with pytest.raises(ValueError):
        MoEConfig(expert_backend="grouped", virtual_block_size=block, grouped_token_budget=budget)


def test_invalid_counts():
    with pytest.raises(ValueError, match="nonnegative"):
        plan_virtual_blocks([2, -1], 1, 8)


def test_grouped_requires_default_dispatch():
    from tests.fixtures.config_fixtures import create_moe_test_config

    from ironcore.layers.moe.moe_layer import CommunicationMode, MoEMLP

    config = create_moe_test_config()
    config.model.moe.expert_backend = "grouped"
    with pytest.raises(ValueError, match="default dispatch"):
        MoEMLP(config, communication_mode=CommunicationMode.ALL_TO_ALL)
