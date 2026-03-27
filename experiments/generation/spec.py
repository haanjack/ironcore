from dataclasses import dataclass, field
from typing import Callable, Optional, Any


@dataclass
class PerformanceTargets:
    """Performance targets for kernel optimization.

    Attributes:
        min_speedup: Minimum speedup over reference to consider successful (default: 2.0x)
        max_time_ms: Maximum execution time in milliseconds (optional)
        min_bandwidth_pct: Minimum percentage of theoretical peak bandwidth (optional)
        min_occupancy: Minimum GPU occupancy percentage (optional)
        custom_targets: Dictionary of custom performance targets
    """
    min_speedup: float = 2.0
    max_time_ms: Optional[float] = None
    min_bandwidth_pct: Optional[float] = None
    min_occupancy: Optional[float] = None
    custom_targets: dict[str, Any] = field(default_factory=dict)

    def is_met(self, validation_result) -> tuple[bool, list[str]]:
        """Check if performance targets are met.

        Args:
            validation_result: ValidationResult from harness

        Returns:
            Tuple of (all_met, list of unmet target descriptions)
        """
        unmet = []

        if validation_result.speedup < self.min_speedup:
            unmet.append(f"Speedup: {validation_result.speedup:.2f}x < {self.min_speedup:.2f}x")

        if self.max_time_ms and validation_result.kernel_time_ms > self.max_time_ms:
            unmet.append(f"Time: {validation_result.kernel_time_ms:.3f}ms > {self.max_time_ms:.3f}ms")

        if self.min_bandwidth_pct:
            bandwidth_pct = validation_result.profiler_metrics.get("bandwidth_pct", 0)
            if bandwidth_pct < self.min_bandwidth_pct:
                unmet.append(f"Bandwidth: {bandwidth_pct:.1f}% < {self.min_bandwidth_pct:.1f}%")

        if self.min_occupancy:
            occupancy = validation_result.profiler_metrics.get("occupancy_pct", 0)
            if occupancy < self.min_occupancy:
                unmet.append(f"Occupancy: {occupancy:.1f}% < {self.min_occupancy:.1f}%")

        for key, target_value in self.custom_targets.items():
            actual_value = validation_result.profiler_metrics.get(key)
            if actual_value is not None and actual_value < target_value:
                unmet.append(f"{key}: {actual_value} < {target_value}")

        return len(unmet) == 0, unmet


@dataclass
class KernelSpec:
    """Specification for a GPU kernel to be implemented and validated.

    Attributes:
        name: Unique identifier for the kernel (e.g. "rmsnorm_forward").
        description: What the kernel computes.
        reference_fn: PyTorch function that produces the correct output.
            Signature: (*inputs) -> output_tensor
        input_factory: Callable that produces test inputs for validation.
            Signature: (dtype, device) -> tuple of tensors
        check_backward: Whether to verify gradient correctness.
        atol: Absolute tolerance for numerical comparison.
        rtol: Relative tolerance for numerical comparison.
        target_file: Path where the kernel implementation should be written,
            relative to the project root.
        kernel_fn_name: Name of the callable to import from target_file.
        input_sizes: List of (description, factory_kwargs) for benchmarking
            at multiple sizes.
        performance_targets: PerformanceTargets for optimization goals.
        optimization_hints: List of specific optimization techniques to consider.
    """
    name: str
    description: str
    reference_fn: Callable
    input_factory: Callable
    check_backward: bool = True
    atol: float = 1e-5
    rtol: float = 1e-5
    target_file: str = ""
    kernel_fn_name: str = ""
    input_sizes: list = field(default_factory=list)
    performance_targets: PerformanceTargets = field(default_factory=PerformanceTargets)
    optimization_hints: list[str] = field(default_factory=list)


# Global registry of kernel specs
_REGISTRY: dict[str, KernelSpec] = {}


def register_spec(spec: KernelSpec):
    """Register a kernel spec in the global registry."""
    _REGISTRY[spec.name] = spec
    return spec


def get_spec(name: str) -> KernelSpec:
    """Get a kernel spec by name."""
    if name not in _REGISTRY:
        available = ", ".join(sorted(_REGISTRY.keys()))
        raise KeyError(f"Unknown kernel spec '{name}'. Available: {available}")
    return _REGISTRY[name]


def list_specs() -> list[str]:
    """List all registered kernel spec names."""
    return sorted(_REGISTRY.keys())
