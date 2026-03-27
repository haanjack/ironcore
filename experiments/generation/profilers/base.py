"""Base profiler interface and result dataclass."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable


@dataclass
class ProfilerResult:
    """Result from profiling a kernel execution.

    Attributes:
        profiler_name: Name of the profiler used
        time_ms: Execution time in milliseconds (wall-clock)
        metrics: Dictionary of additional metrics (GPU time, memory bandwidth, etc.)
        output_path: Path to profiler output file (if any)
        raw_output: Raw profiler output (for debugging/parsing)
        success: Whether profiling succeeded
        error_msg: Error message if profiling failed
    """
    profiler_name: str
    time_ms: float
    metrics: dict[str, Any] = field(default_factory=dict)
    output_path: Path | None = None
    raw_output: str = ""
    success: bool = True
    error_msg: str = ""

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "profiler_name": self.profiler_name,
            "time_ms": self.time_ms,
            "metrics": self.metrics,
            "output_path": str(self.output_path) if self.output_path else None,
            "raw_output": self.raw_output,
            "success": self.success,
            "error_msg": self.error_msg,
        }


class Profiler(ABC):
    """Abstract base class for kernel profilers."""

    # Warmup iterations before profiling
    warmup: int = 10

    # Number of profiled iterations
    repeat: int = 100

    # Output directory for profiler files
    output_dir: Path

    def __init__(self, output_dir: Path, warmup: int = 10, repeat: int = 100):
        """Initialize profiler.

        Args:
            output_dir: Directory to store profiler outputs
            warmup: Number of warmup iterations
            repeat: Number of profiled iterations
        """
        self.output_dir = Path(output_dir)
        self.warmup = warmup
        self.repeat = repeat
        self.output_dir.mkdir(parents=True, exist_ok=True)

    @abstractmethod
    def is_available(self) -> bool:
        """Check if this profiler is available on the system.

        Returns:
            True if profiler can be used, False otherwise
        """
        ...

    @abstractmethod
    def profile(
        self,
        fn: Callable,
        inputs: tuple,
        label: str,
    ) -> ProfilerResult:
        """Profile a function with given inputs.

        Args:
            fn: Function to profile
            inputs: Input arguments for the function
            label: Label for this profiling run (used for filenames)

        Returns:
            ProfilerResult with timing and metrics
        """
        ...

    @abstractmethod
    def parse_results(self, output_path: Path) -> dict[str, Any]:
        """Parse profiler output file to extract metrics.

        Args:
            output_path: Path to profiler output file

        Returns:
            Dictionary of parsed metrics
        """
        ...

    def _do_warmup(self, fn: Callable, inputs: tuple) -> None:
        """Run warmup iterations."""
        import torch
        for _ in range(self.warmup):
            fn(*inputs)
        torch.cuda.synchronize()
