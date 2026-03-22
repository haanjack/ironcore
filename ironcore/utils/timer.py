# Copyright (c) 2025-2026 Jaegeun Han
#
# SPDX-License-Identifier: Apache-2.0

import time


class Timer:
    """Timer"""

    def __init__(self):
        self.timers: dict[str, list[float]] = {}
        self.running: dict[str, bool] = {}

    def register(self, name: str):
        """Register timer."""
        if name in self.running:
            raise KeyError(f"Requested timer ({name}) is already registered")
        self.timers[name] = []
        self.running[name] = False

    def start(self, name: str):
        """Start timer."""
        if name not in self.running:
            self.register(name)

        assert not self.running[name], (
            f"Timer {name} is already running. This can happen in duplicated operaiton"
        )

        self.running[name] = True
        self.timers[name].append(time.time())

    def stop(self, name: str):
        """Stop timer."""
        if name not in self.running:
            raise KeyError(f"Not initialized timer ({name}) is requested")
        if not self.running[name]:
            raise RuntimeError(
                f"Stopping timer {name} is requested while it is already stopped. This can be duplicated operation."
            )

        self.running[name] = False
        self.timers[name][-1] = time.time() - self.timers[name][-1]

    def get(self, name: str) -> float:
        """Get summary of requested timer."""
        if name not in self.running:
            raise KeyError(f"Not initialized timer ({name}) is requested")
        if self.running[name]:
            self.stop(name)
        return sum(self.timers[name]) / len(self.timers[name])

    def get_summary(self) -> dict[str, float]:
        """Get summary of all timers."""
        summary = {}
        for name, times in self.timers.items():
            if len(times) == 0:
                continue
            summary[name] = sum(times) / len(times)
        return summary

    def reset(self, name: str):
        """Reset timer."""
        if name not in self.running:
            raise KeyError(f"Not initialized timer ({name}) is requested")
        self.timers[name] = []
        self.running[name] = False

    def reset_all(self):
        """Reset all timers."""
        for name in self.timers.keys():
            self.reset(name)

    def stop_all(self):
        """Stop all timers."""
        for name in self.timers.keys():
            self.stop(name)
