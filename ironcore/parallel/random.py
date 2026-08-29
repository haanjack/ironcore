# Copyright (c) 2025-2026 Jaegeun Han
#
# SPDX-License-Identifier: Apache-2.0

"""Independent RNG streams for tensor-parallel execution."""

from collections.abc import Iterator
from contextlib import contextmanager

import torch

from ironcore.parallel import parallel_states


class TensorParallelRNGTracker:
    """Track RNG states that must stay synchronized inside a TP group."""

    def __init__(self) -> None:
        self._states: dict[tuple[str, int, str, int | None], torch.Tensor] = {}

    def reset(self) -> None:
        self._states.clear()

    def snapshot(self) -> dict[tuple[str, int, str, int | None], torch.Tensor]:
        """Copy every tracked stream position."""
        return {key: state.clone() for key, state in self._states.items()}

    def restore(self, states: dict[tuple[str, int, str, int | None], torch.Tensor]) -> None:
        """Rewind to exactly the positions captured by `snapshot`.

        Replaces rather than merges. A key missing from the snapshot means the
        stream had not been seeded yet at that point, so it must go back to not
        existing — merging would leave it wherever a later `fork` advanced it,
        and the next draw would continue that stream instead of the fresh
        lazily-seeded one the original run made.
        """
        self._states.clear()
        self._states.update({key: state.clone() for key, state in states.items()})

    @staticmethod
    def _ranks() -> tuple[int, int]:
        try:
            dp_rank = parallel_states.get_data_parallel_group_rank()
        except RuntimeError:
            dp_rank = 0
        try:
            tp_rank = parallel_states.get_tensor_model_parallel_rank()
        except RuntimeError:
            tp_rank = 0
        return dp_rank, tp_rank

    @contextmanager
    def fork(
        self,
        seed: int,
        device: torch.device,
        *,
        sharded: bool = False,
    ) -> Iterator[None]:
        """Run an operation using a tracked replicated or sharded RNG stream."""
        device = torch.device(device)
        dp_rank, tp_rank = self._ranks()
        stream = "sharded" if sharded else "replicated"
        effective_seed = seed + dp_rank * 100_003 + (tp_rank if sharded else 0)
        key = (stream, effective_seed, device.type, device.index)

        if device.type == "cuda":

            def get_state():
                return torch.cuda.get_rng_state(device)

            def set_state(state):
                torch.cuda.set_rng_state(state, device=device)
        else:
            get_state = torch.get_rng_state
            set_state = torch.set_rng_state

        if key not in self._states:
            generator = torch.Generator(device=device)
            generator.manual_seed(effective_seed)
            self._states[key] = generator.get_state()

        original_state = get_state()
        set_state(self._states[key])
        try:
            yield
        finally:
            self._states[key] = get_state()
            set_state(original_state)


_TP_RNG_TRACKER = TensorParallelRNGTracker()


def reset_tensor_parallel_rng_tracker() -> None:
    _TP_RNG_TRACKER.reset()


def snapshot_tensor_parallel_rng_tracker() -> dict[tuple[str, int, str, int | None], torch.Tensor]:
    """Capture the shared tracker's stream positions."""
    return _TP_RNG_TRACKER.snapshot()


@contextmanager
def tensor_parallel_rng_rewound_to(
    states: dict[tuple[str, int, str, int | None], torch.Tensor],
) -> Iterator[None]:
    """Run a block with the shared tracker rewound to `states`, then put it back.

    The counterpart to `torch.random.fork_rng` for the tracker's streams, which
    live outside the ambient generators and so survive that context manager
    untouched. Activation spill and gradient checkpointing need this: their
    backward recompute must redraw the dropout masks its forward drew, which
    means rewinding — but the rewind has to be undone afterwards, or the next
    microbatch's forward would replay masks instead of continuing the stream.
    """
    saved = _TP_RNG_TRACKER.snapshot()
    _TP_RNG_TRACKER.restore(states)
    try:
        yield
    finally:
        _TP_RNG_TRACKER.restore(saved)


@contextmanager
def tensor_parallel_rng_fork(
    seed: int,
    device: torch.device,
    *,
    sharded: bool = False,
) -> Iterator[None]:
    """Fork the shared tensor-parallel RNG tracker."""
    with _TP_RNG_TRACKER.fork(seed, device, sharded=sharded):
        yield
