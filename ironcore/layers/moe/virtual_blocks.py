# Copyright (c) 2025-2026 Jaegeun Han
# SPDX-License-Identifier: Apache-2.0

"""Logical expert token tiles and bounded grouped-GEMM execution plans."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class VirtualBlock:
    """A view of sorted routed tokens owned by one expert; no tensor allocation."""

    expert: int
    start: int
    tokens: int


@dataclass(frozen=True)
class ExecutionGroup:
    """Consecutive virtual tiles coalesced into jagged expert GEMMs."""

    start: int
    tokens: int
    experts: tuple[int, ...]
    counts: tuple[int, ...]

    @property
    def offsets(self) -> tuple[int, ...]:
        """Exclusive group endpoints required by grouped_mm."""
        total = 0
        result = []
        for count in self.counts:
            total += count
            result.append(total)
        return tuple(result)


@dataclass(frozen=True)
class VirtualBlockPlan:
    """Logical ownership is independent of the physical GEMM token budget."""

    blocks: tuple[VirtualBlock, ...]
    groups: tuple[ExecutionGroup, ...]


def plan_virtual_blocks(
    expert_counts: list[int], block_size: int, token_budget: int
) -> VirtualBlockPlan:
    """Cover every routed token exactly once without padded expert capacities.

    Adjacent logical tiles of an expert become one GEMM segment. Execution
    groups may include several experts; the sum of their rows never exceeds
    token_budget. A partial final tile consumes only its valid rows.
    """
    if not isinstance(block_size, int) or isinstance(block_size, bool) or block_size < 1:
        raise ValueError("virtual_block_size must be a positive integer")
    if (
        not isinstance(token_budget, int)
        or isinstance(token_budget, bool)
        or token_budget < block_size
    ):
        raise ValueError("grouped_token_budget must be an integer >= virtual_block_size")
    if any(not isinstance(n, int) or isinstance(n, bool) or n < 0 for n in expert_counts):
        raise ValueError("expert token counts must be nonnegative integers")
    blocks = []
    position = 0
    for expert, count in enumerate(expert_counts):
        for offset in range(0, count, block_size):
            blocks.append(VirtualBlock(expert, position + offset, min(block_size, count - offset)))
        position += count

    groups = []
    current = []
    rows = 0

    def finish() -> None:
        nonlocal current, rows
        if not current:
            return
        experts, counts = [], []
        for block in current:
            if experts and experts[-1] == block.expert:
                counts[-1] += block.tokens
            else:
                experts.append(block.expert)
                counts.append(block.tokens)
        groups.append(ExecutionGroup(current[0].start, rows, tuple(experts), tuple(counts)))
        current, rows = [], 0

    for block in blocks:
        if rows + block.tokens > token_budget:
            finish()
        current.append(block)
        rows += block.tokens
    finish()
    return VirtualBlockPlan(tuple(blocks), tuple(groups))
