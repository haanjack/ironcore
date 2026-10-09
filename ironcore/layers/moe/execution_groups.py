# Copyright (c) 2025-2026 Jaegeun Han
# SPDX-License-Identifier: Apache-2.0

"""Bounded grouped-GEMM plans built directly from sorted expert counts."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ExecutionGroup:
    """A contiguous routed-token range with one GEMM segment per expert."""

    start: int
    tokens: int
    experts: tuple[int, ...]
    counts: tuple[int, ...]

    @property
    def offsets(self) -> tuple[int, ...]:
        """Exclusive segment endpoints for grouped_mm."""
        total = 0
        result = []
        for count in self.counts:
            total += count
            result.append(total)
        return tuple(result)


def plan_execution_groups(
    expert_counts: list[int], token_budget: int
) -> tuple[ExecutionGroup, ...]:
    """Cover sorted assignments once, filling every group except the final tail.

    Split an expert at a group boundary and fill spare rows with the next expert.
    Planning cost depends on expert and execution-group counts, not token tiles.
    Zero-token experts have no segment and retain their original expert ids.
    """
    if not isinstance(token_budget, int) or isinstance(token_budget, bool) or token_budget < 1:
        raise ValueError("grouped_token_budget must be a positive integer")
    if any(not isinstance(n, int) or isinstance(n, bool) or n < 0 for n in expert_counts):
        raise ValueError("expert token counts must be nonnegative integers")
    groups = []
    start, rows = 0, 0
    owners, counts = [], []
    for expert, remaining in enumerate(expert_counts):
        while remaining:
            take = min(remaining, token_budget - rows)
            owners.append(expert)
            counts.append(take)
            rows += take
            remaining -= take
            if rows == token_budget:
                groups.append(ExecutionGroup(start, rows, tuple(owners), tuple(counts)))
                start += rows
                rows = 0
                owners, counts = [], []
    if rows:
        groups.append(ExecutionGroup(start, rows, tuple(owners), tuple(counts)))
    return tuple(groups)
