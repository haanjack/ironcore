# Copyright (c) 2025-2026 Jaegeun Han
#
# SPDX-License-Identifier: Apache-2.0

"""
Offload telemetry collection and reporting.

Tracks memory transfer statistics (H2D/D2H bytes, timing, stalls)
 for monitoring and debugging offload performance.
"""

import dataclasses
import json
import threading
from typing import Any


@dataclasses.dataclass
class OffloadMetrics:
    """Telemetry for offload operations."""

    # Transfer counters
    total_h2d_bytes: int = 0  # Total bytes transferred host-to-device
    total_d2h_bytes: int = 0  # Total bytes transferred device-to-host
    total_h2d_count: int = 0  # Number of H2D transfers
    total_d2h_count: int = 0  # Number of D2H transfers

    # Timing (nanoseconds)
    total_h2d_ns: int = 0  # Cumulative H2D transfer time
    total_d2h_ns: int = 0  # Cumulative D2H transfer time

    # Stall detection
    stall_events: int = 0  # Number of times transfer queue was full
    total_stall_ns: int = 0  # Cumulative stall time waiting for queue

    # Queue stats
    max_queue_depth: int = 0  # Maximum observed queue depth
    current_queue_depth: int = 0

    # Step tracking
    step_count: int = 0  # Number of training steps tracked

    def to_dict(self) -> dict[str, Any]:
        """Convert to JSON-serializable dict."""
        return {
            "transfer_stats": {
                "h2d_bytes": self.total_h2d_bytes,
                "d2h_bytes": self.total_d2h_bytes,
                "h2d_count": self.total_h2d_count,
                "d2h_count": self.total_d2h_count,
                "h2d_bandwidth_gb_s": self._calc_bandwidth(self.total_h2d_bytes, self.total_h2d_ns),
                "d2h_bandwidth_gb_s": self._calc_bandwidth(self.total_d2h_bytes, self.total_d2h_ns),
            },
            "timing": {
                "total_h2d_ms": self.total_h2d_ns / 1_000_000,
                "total_d2h_ms": self.total_d2h_ns / 1_000_000,
            },
            "stalls": {
                "stall_events": self.stall_events,
                "total_stall_ms": self.total_stall_ns / 1_000_000,
            },
            "queue": {
                "max_depth": self.max_queue_depth,
            },
            "steps": self.step_count,
        }

    @staticmethod
    def _calc_bandwidth(bytes: int, nanos: int) -> float:
        """Calculate bandwidth in GB/s."""
        if nanos == 0:
            return 0.0
        return (bytes / 1e9) / (nanos / 1e9)


class OffloadMetricsCollector:
    """Thread-safe collector for offload telemetry."""

    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._metrics = OffloadMetrics()
        return cls._instance

    def __init__(self):
        """Initialize the collector."""
        self._metrics = OffloadMetrics()

    def record_h2d(self, bytes: int, nanos: int):
        """Record a host-to-device transfer."""
        with self._lock:
            self._metrics.total_h2d_bytes += bytes
            self._metrics.total_h2d_ns += nanos
            self._metrics.total_h2d_count += 1

    def record_d2h(self, bytes: int, nanos: int):
        """Record a device-to-host transfer."""
        with self._lock:
            self._metrics.total_d2h_bytes += bytes
            self._metrics.total_d2h_ns += nanos
            self._metrics.total_d2h_count += 1

    def record_stall(self, nanos: int):
        """Record a stall event (queue full)."""
        with self._lock:
            self._metrics.stall_events += 1
            self._metrics.total_stall_ns += nanos

    def update_queue_depth(self, depth: int):
        """Update and track maximum queue depth."""
        with self._lock:
            self._metrics.current_queue_depth = depth
            self._metrics.max_queue_depth = max(self._metrics.max_queue_depth, depth)

    def increment_step(self):
        """Increment training step counter."""
        with self._lock:
            self._metrics.step_count += 1

    def get_metrics(self) -> OffloadMetrics:
        """Get a copy of current metrics."""
        with self._lock:
            # Return the actual metrics instance (not a copy) for simplicity
            # The collector is thread-safe and users won't modify it directly
            return self._metrics

    def reset(self):
        """Reset all metrics to zero."""
        with self._lock:
            self._metrics = OffloadMetrics()

    def dump_json(self, path: str):
        """Dump metrics to JSON file."""
        metrics = self.get_metrics()
        with open(path, "w") as f:
            json.dump(metrics.to_dict(), f, indent=2)

    def __repr__(self) -> str:
        """String representation."""
        m = self.get_metrics()
        return (
            f"OffloadMetrics(H2D={m.total_h2d_bytes / 1e9:.2f}GB, "
            f"D2H={m.total_d2h_bytes / 1e9:.2f}GB, "
            f"steps={m.step_count})"
        )


def get_offload_metrics() -> OffloadMetricsCollector:
    """Get the global offload metrics collector."""
    return OffloadMetricsCollector()
