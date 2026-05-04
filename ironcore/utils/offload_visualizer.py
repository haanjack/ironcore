#!/usr/bin/env python3
"""
Simple offload telemetry visualizer for training.

Displays periodic updates of offload metrics during training.
Uses only standard library - no external dependencies.

Usage:
    # In your training script
    from ironcore.utils.offload_visualizer import start_offload_visualizer

    viz = start_offload_visualizer(update_interval=10)
    # ... training loop ...
    viz.stop()
"""

import sys
import threading
import time

try:
    from ironcore.utils.offload_metrics import OffloadMetrics, get_offload_metrics
except ImportError:
    print("Warning: offload_metrics not available. Install ironcore.")
    sys.exit(0)


class OffloadVisualizer:
    """Terminal-based offload telemetry visualizer."""

    def __init__(self, update_interval: int = 10):
        """Initialize the visualizer.

        Args:
            update_interval: Steps between display updates
        """
        self.update_interval = update_interval
        self._running = False
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._last_step = -1

    def _format_metrics(self, metrics: OffloadMetrics) -> str:
        """Format metrics as a terminal table."""
        h2d_gb = metrics.total_h2d_bytes / 1e9
        d2h_gb = metrics.total_d2h_bytes / 1e9
        stats = metrics.to_dict()
        h2d_bw = stats["transfer_stats"]["h2d_bandwidth_gb_s"]
        d2h_bw = stats["transfer_stats"]["d2h_bandwidth_gb_s"]

        return f"""
{"=" * 60}
[Offload Telemetry] Step {metrics.step_count}
{"─" * 60}
H2D: {h2d_gb:.2f} GB | {h2d_bw:.2f} GB/s | {metrics.total_h2d_count} transfers
D2H: {d2h_gb:.2f} GB | {d2h_bw:.2f} GB/s | {metrics.total_d2h_count} transfers
Stalls: {metrics.stall_events} events | {stats["timing"]["total_h2d_ms"] + stats["timing"]["total_d2h_ms"]:.1f} ms total
Queue: {metrics.current_queue_depth}/{metrics.max_queue_depth} depth
{"=" * 60}"""

    def _update_loop(self):
        """Background thread that updates the display."""
        while self._running:
            time.sleep(self.update_interval / 10)  # Check frequently

            with self._lock:
                if not self._running:
                    break

            metrics = get_offload_metrics().get_metrics()

            # Only update if step count changed
            if metrics.step_count > self._last_step:
                print(self._format_metrics(metrics))
                self._last_step = metrics.step_count

    def start(self):
        """Start the visualizer in a background thread."""
        with self._lock:
            if self._running:
                return
            self._running = True

        self._thread = threading.Thread(target=self._update_loop, daemon=True)
        self._thread.start()
        print(f"[OffloadVisualizer] Started (updates every {self.update_interval} steps)")

    def stop(self):
        """Stop the visualizer and print final summary."""
        with self._lock:
            self._running = False

        if self._thread:
            self._thread.join(timeout=2)

        # Print final summary
        metrics = get_offload_metrics().get_metrics()
        print(f"\n{'=' * 60}")
        print("[bold] Final Offload Summary[/bold]")
        print(self._format_metrics(metrics))

    def snapshot(self) -> str:
        """Return a snapshot of current metrics as a string."""
        metrics = get_offload_metrics().get_metrics()
        return self._format_metrics(metrics)


def start_offload_visualizer(update_interval: int = 10) -> OffloadVisualizer:
    """Start the offload telemetry visualizer.

    Args:
        update_interval: Update display every N training steps

    Returns:
        OffloadVisualizer instance (call stop() when done)

    Example:
        viz = start_offload_visualizer()
        try:
            trainer.train()
        finally:
            viz.stop()
    """
    viz = OffloadVisualizer(update_interval=update_interval)
    viz.start()
    return viz


def print_offload_snapshot():
    """Print a one-time snapshot of current offload metrics."""
    viz = SimpleOffloadVisualizer()
    print(viz.snapshot())
