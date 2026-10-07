# Copyright (c) 2025-2026 Jaegeun Han
# SPDX-License-Identifier: Apache-2.0
"""Iterator ownership for consumed-batch state, including worker prefetch."""


class CheckpointableIterator:
    def __init__(self, loader, cycle=False):
        self.loader = loader
        self.iterator = None
        self.cycle = cycle
        self.epoch = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.iterator is None:
            self.iterator = iter(self.loader)
        try:
            return next(self.iterator)
        except StopIteration:
            if not self.cycle:
                raise
            self.epoch += 1
            self.iterator = iter(self.loader)
            try:
                return next(self.iterator)
            except StopIteration as error:
                raise ValueError(
                    "Training dataset has no samples for this DP rank/worker configuration"
                ) from error

    def state_dict(self):
        return {
            "epoch": self.epoch,
            "num_workers": self.loader.num_workers,
            "batch_size": self.loader.batch_size,
            "loader": self.loader.state_dict(),
        }

    def load_state_dict(self, state):
        if (state["num_workers"], state["batch_size"]) != (
            self.loader.num_workers,
            self.loader.batch_size,
        ):
            raise ValueError("Data resume requires the original worker count and microbatch size")
        self.epoch = state["epoch"]
        self.loader.load_state_dict(state["loader"])
        self.iterator = iter(self.loader)
