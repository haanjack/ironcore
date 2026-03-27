"""Profiler plugins for kernel benchmarking.

Supported profilers:
- torch: PyTorch built-in profiler (DEFAULT, captures GPU metrics)
- wallclock: Simple timing only (fastest)

PyTorch profiler advantages:
- Cross-platform: Works on NVIDIA (CUDA) and AMD (ROCm/HIP)
- No external dependencies required
- Captures actual GPU kernel execution time
- Exports Chrome trace format for visualization
- Reliable for automated testing loops

Usage:
    # Default: torch profiler (captures GPU metrics)
    python -m experiments.generation.harness rmsnorm

    # Simple timing only (no GPU metrics)
    python -m experiments.generation.harness rmsnorm --profiler wallclock

AI-Based Analysis:
    # Use AI to analyze profiling results
    from experiments.generation.profilers.analyzer import ProfilingAnalyzer
    analyzer = ProfilingAnalyzer(provider_name="glm")
    analysis = analyzer.analyze_results(spec, validation_result, current_code)
"""

from experiments.generation.profilers.base import Profiler, ProfilerResult
from experiments.generation.profilers.wallclock import WallClockProfiler
from experiments.generation.profilers.torch_profiler import TorchProfiler

__all__ = [
    "Profiler",
    "ProfilerResult",
    "WallClockProfiler",
    "TorchProfiler",
    "get_profiler",
    "list_profilers",
    "DEFAULT_PROFILER",
    "ProfilingAnalyzer",
    "OptimizationAnalysis",
    "get_analyzer",
]

# Default profiler for automated testing
DEFAULT_PROFILER = "torch"


_PROFILER_REGISTRY: dict[str, type[Profiler]] = {
    "wallclock": WallClockProfiler,
    "torch": TorchProfiler,
}


def get_profiler(name: str, **kwargs) -> Profiler:
    """Get a profiler instance by name.

    Args:
        name: Profiler name (torch or wallclock)
        **kwargs: Additional arguments passed to profiler constructor

    Returns:
        Profiler instance

    Raises:
        ValueError: If profiler name is unknown
    """
    if name not in _PROFILER_REGISTRY:
        available = ", ".join(sorted(_PROFILER_REGISTRY.keys()))
        raise ValueError(f"Unknown profiler '{name}'. Available: {available}")
    return _PROFILER_REGISTRY[name](**kwargs)


def list_profilers() -> list[str]:
    """List available profiler names."""
    return sorted(_PROFILER_REGISTRY.keys())


# Lazy import for analyzer to avoid circular dependencies
def get_analyzer(provider_name: str = "openai", model: str = None,
                api_key: str = None, base_url: str = None):
    """Get a profiling analyzer instance.

    Args:
        provider_name: AI provider name
        model: Model name (uses provider default if None)
        api_key: API key
        base_url: Optional custom base URL

    Returns:
        ProfilingAnalyzer instance
    """
    from experiments.generation.profilers.analyzer import ProfilingAnalyzer, get_analyzer as _get_analyzer
    return _get_analyzer(provider_name, model, api_key, base_url)


def ProfilingAnalyzer(*args, **kwargs):
    """Proxy for ProfilingAnalyzer class."""
    from experiments.generation.profilers.analyzer import ProfilingAnalyzer as _ProfilerAnalyzer
    return _ProfilerAnalyzer(*args, **kwargs)


def OptimizationAnalysis(*args, **kwargs):
    """Proxy for OptimizationAnalysis class."""
    from experiments.generation.profilers.analyzer import OptimizationAnalysis as _OptimizationAnalysis
    return _OptimizationAnalysis(*args, **kwargs)
