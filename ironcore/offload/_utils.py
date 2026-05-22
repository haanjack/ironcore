# Copyright (c) 2025-2026 Jaegeun Han
#
# SPDX-License-Identifier: Apache-2.0
"""Shared low-level helpers for offload pool implementations."""

from __future__ import annotations

import torch

_ELEMENT_SIZES: dict[torch.dtype, int] = {
    torch.float32: 4,
    torch.float16: 2,
    torch.bfloat16: 2,
    torch.int64: 8,
    torch.int32: 4,
    torch.uint8: 1,
}


def _element_size(dtype: torch.dtype) -> int:
    return _ELEMENT_SIZES.get(dtype) or torch.tensor([], dtype=dtype).element_size()


def _coalesce_free_list(
    free_list: list[tuple[int, int]],
    offset: int,
    num_bytes: int,
) -> list[tuple[int, int]]:
    """
    Merge a freed region into an existing free list.

    Iterates to fixed-point so bridging merges (freed region spans two existing
    entries) are handled correctly in a single call.

    Returns the updated free list with the freed region merged in.
    """
    merged_start = offset
    merged_end = offset + num_bytes

    changed = True
    while changed:
        changed = False
        remaining: list[tuple[int, int]] = []
        for region_start, region_numel in free_list:
            region_end = region_start + region_numel
            if region_end == merged_start:
                merged_start = region_start
                changed = True
            elif region_start == merged_end:
                merged_end = region_end
                changed = True
            else:
                remaining.append((region_start, region_numel))
        free_list = remaining

    free_list.append((merged_start, merged_end - merged_start))
    return free_list
