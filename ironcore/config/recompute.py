# Copyright (c) 2025-2026 Jaegeun Han
# SPDX-License-Identifier: Apache-2.0
"""Plan checkpoint input boundaries independently of their storage device."""


def recompute_segments(
    num_layers: int, interval: int = 1, skip_count: int | None = None
) -> tuple[tuple[int, int], ...]:
    """Return contiguous [start, end) groups, preserving the first input.

    Count mode evenly distributes exactly N - skip_count stored inputs. Segment
    lengths differ by at most one; adjacent omissions are unavoidable when more
    than half of the boundaries are omitted. None selects uniform interval mode.
    """
    if isinstance(interval, bool) or not isinstance(interval, int) or interval < 0:
        raise ValueError("operation.recompute_interval must be a nonnegative integer")
    if skip_count is not None:
        if (
            isinstance(skip_count, bool)
            or not isinstance(skip_count, int)
            or not 0 <= skip_count < num_layers
        ):
            raise ValueError("operation.recompute_skip_count must be an integer in [0, num_layers)")
        if interval != 1:
            raise ValueError("recompute_skip_count requires recompute_interval=1 (the default)")
        groups = num_layers - skip_count
        boundaries = [(i * num_layers + groups - 1) // groups for i in range(groups + 1)]
        return tuple(zip(boundaries[:-1], boundaries[1:], strict=True))
    if interval == 0:
        return ()
    return tuple(
        (start, min(start + interval, num_layers)) for start in range(0, num_layers, interval)
    )
