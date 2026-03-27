"""Validation and benchmark harness for DSL kernel development.

Usage:
    # Validate a specific kernel
    python -m experiments.generation.harness rmsnorm

    # Validate all registered kernels
    python -m experiments.generation.harness --all

    # Benchmark only (skip correctness checks)
    python -m experiments.generation.harness rmsnorm --benchmark-only

    # Verbose output
    python -m experiments.generation.harness rmsnorm --verbose

    # Use specific profiler
    python -m experiments.generation.harness rmsnorm --profiler nsys
    python -m experiments.generation.harness rmsnorm --profiler rocprofv3

    # List available profilers
    python -m experiments.generation.harness --list-profilers
"""

import argparse
import importlib
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

from experiments.generation.spec import KernelSpec, get_spec, list_specs
from experiments.generation.profilers import get_profiler, list_profilers, ProfilerResult, DEFAULT_PROFILER

# Import all spec modules to trigger registration
import experiments.generation.specs.rmsnorm  # noqa: F401
import experiments.generation.specs.layernorm  # noqa: F401
import experiments.generation.specs.softmax  # noqa: F401


PROJECT_ROOT = Path(__file__).parent.parent.parent
RESULTS_DIR = Path(__file__).parent / "results"
PROFILER_DIR = RESULTS_DIR / "profiler"


@dataclass
class ValidationResult:
    name: str
    correct: bool
    grad_correct: bool
    ref_time_ms: float
    kernel_time_ms: float
    speedup: float
    error_msg: str = ""
    max_abs_diff: float = 0.0
    max_rel_diff: float = 0.0
    profiler_name: str = "wallclock"
    profiler_metrics: dict[str, Any] = field(default_factory=dict)
    profiler_output: str | None = None


def _load_kernel_fn(spec: KernelSpec):
    """Dynamically import the kernel function from the target file."""
    target = PROJECT_ROOT / spec.target_file
    if not target.exists():
        raise FileNotFoundError(
            f"Kernel file not found: {spec.target_file}\n"
            f"Expected at: {target}\n"
            f"Implement the kernel and place it there."
        )

    # Convert file path to module path
    # e.g. "ironcore/kernels/triton/rmsnorm.py" -> "ironcore.kernels.triton.rmsnorm"
    module_path = spec.target_file.replace("/", ".").removesuffix(".py")
    module = importlib.import_module(module_path)

    fn = getattr(module, spec.kernel_fn_name, None)
    if fn is None:
        raise AttributeError(
            f"Function '{spec.kernel_fn_name}' not found in {spec.target_file}.\n"
            f"Available: {[x for x in dir(module) if not x.startswith('_')]}"
        )
    return fn


def write_kernel(spec: KernelSpec, code: str):
    """Write generated kernel code to the target file.

    Args:
        spec: Kernel specification containing target_file path
        code: Generated kernel code to write
    """
    target = PROJECT_ROOT / spec.target_file
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(code)


def check_correctness(spec: KernelSpec, kernel_fn, dtype, device, verbose=False):
    """Check numerical correctness of kernel against reference."""
    inputs = spec.input_factory(dtype=dtype, device=device)

    with torch.no_grad():
        ref_output = spec.reference_fn(*inputs)
        kernel_output = kernel_fn(*inputs)

    if ref_output.shape != kernel_output.shape:
        return False, f"Shape mismatch: ref={ref_output.shape} kernel={kernel_output.shape}", 0.0, 0.0

    abs_diff = (ref_output - kernel_output).abs()
    max_abs = abs_diff.max().item()

    # Avoid division by zero in relative diff
    ref_abs = ref_output.abs().clamp(min=1e-12)
    max_rel = (abs_diff / ref_abs).max().item()

    correct = torch.allclose(kernel_output, ref_output, atol=spec.atol, rtol=spec.rtol)

    if verbose:
        print(f"  Max absolute diff: {max_abs:.2e}")
        print(f"  Max relative diff: {max_rel:.2e}")
        print(f"  Tolerance: atol={spec.atol:.0e}, rtol={spec.rtol:.0e}")

    error_msg = ""
    if not correct:
        error_msg = f"Numerical mismatch: max_abs={max_abs:.2e}, max_rel={max_rel:.2e}"

    return correct, error_msg, max_abs, max_rel


def check_gradient(spec: KernelSpec, kernel_fn, dtype, device, verbose=False):
    """Check gradient correctness via comparison with reference backward pass."""
    if not spec.check_backward:
        return True, ""

    # Use float32 for gradient checking (more numerically stable)
    grad_dtype = torch.float32
    inputs_ref = spec.input_factory(dtype=grad_dtype, device=device)
    inputs_kernel = tuple(
        x.clone().detach().requires_grad_(x.requires_grad) if isinstance(x, torch.Tensor) and x.is_floating_point()
        else x
        for x in inputs_ref
    )

    # Mark inputs requiring grad
    ref_grad_inputs = []
    kernel_grad_inputs = []
    for r, k in zip(inputs_ref, inputs_kernel):
        if isinstance(r, torch.Tensor) and r.is_floating_point() and r.requires_grad:
            ref_grad_inputs.append(r)
            kernel_grad_inputs.append(k)

    if not ref_grad_inputs:
        if verbose:
            print("  No gradient inputs found, skipping gradient check")
        return True, ""

    ref_output = spec.reference_fn(*inputs_ref)
    kernel_output = kernel_fn(*inputs_kernel)

    # Backward with ones
    grad_out = torch.ones_like(ref_output)
    ref_output.backward(grad_out)
    kernel_output.backward(grad_out.clone())

    for i, (rg, kg) in enumerate(zip(ref_grad_inputs, kernel_grad_inputs)):
        if rg.grad is None or kg.grad is None:
            return False, f"Gradient is None for input {i}"

        grad_close = torch.allclose(kg.grad, rg.grad, atol=spec.atol * 10, rtol=spec.rtol * 10)
        if verbose:
            diff = (kg.grad - rg.grad).abs().max().item()
            print(f"  Gradient input {i}: max_diff={diff:.2e}, pass={grad_close}")

        if not grad_close:
            diff = (kg.grad - rg.grad).abs().max().item()
            return False, f"Gradient mismatch for input {i}: max_diff={diff:.2e}"

    return True, ""


def benchmark_with_profiler(profiler, fn, inputs, label: str) -> ProfilerResult:
    """Benchmark a function using the configured profiler."""
    return profiler.profile(fn, inputs, label=label)


def print_profiler_metrics(metrics: dict[str, Any], indent: str = "    ") -> None:
    """Print profiler metrics in a readable format."""
    for key, value in sorted(metrics.items()):
        if isinstance(value, float):
            # Format floats nicely
            if abs(value) < 1e-3 or abs(value) > 1e6:
                print(f"{indent}{key}: {value:.2e}")
            else:
                print(f"{indent}{key}: {value:.4f}")
        else:
            print(f"{indent}{key}: {value}")


def validate_kernel(
    spec: KernelSpec,
    profiler_name: str = "wallclock",
    verbose: bool = False,
    benchmark_only: bool = False,
) -> ValidationResult:
    """Run full validation of a kernel: correctness, gradients, benchmark.

    Args:
        spec: Kernel specification to validate
        profiler_name: Name of profiler to use (wallclock, nsys, rocprofv3)
        verbose: Enable verbose output
        benchmark_only: Skip correctness checks, only benchmark

    Returns:
        ValidationResult with all results
    """
    device = "cuda"
    dtype = torch.float32

    # Setup profiler
    profiler_output_dir = PROFILER_DIR / spec.name
    profiler = get_profiler(profiler_name, output_dir=profiler_output_dir)

    if not profiler.is_available():
        print(f"  WARNING: Profiler '{profiler_name}' is not available.")
        print(f"  Falling back to 'wallclock' profiler.")
        profiler = get_profiler("wallclock", output_dir=profiler_output_dir)
        profiler_name = "wallclock"

    print(f"\n{'='*60}")
    print(f"Validating: {spec.name}")
    print(f"  Description: {spec.description}")
    print(f"  Target: {spec.target_file}::{spec.kernel_fn_name}")
    print(f"  Profiler: {profiler_name}")
    print(f"{'='*60}")

    # Load kernel
    try:
        kernel_fn = _load_kernel_fn(spec)
    except (FileNotFoundError, AttributeError) as e:
        print(f"  SKIP: {e}")
        return ValidationResult(
            name=spec.name,
            correct=False,
            grad_correct=False,
            ref_time_ms=0,
            kernel_time_ms=0,
            speedup=0,
            error_msg=str(e),
            profiler_name=profiler_name,
        )

    correct = True
    error_msg = ""
    max_abs = 0.0
    max_rel = 0.0
    grad_correct = True

    if not benchmark_only:
        # Correctness
        print("\n  [Correctness]")
        correct, error_msg, max_abs, max_rel = check_correctness(
            spec, kernel_fn, dtype, device, verbose
        )
        status = "PASS" if correct else "FAIL"
        print(f"  Result: {status}")
        if error_msg:
            print(f"  Error: {error_msg}")

        # Gradient
        print("\n  [Gradient]")
        if spec.check_backward:
            grad_correct, grad_error = check_gradient(
                spec, kernel_fn, dtype, device, verbose
            )
            grad_status = "PASS" if grad_correct else "FAIL"
            print(f"  Result: {grad_status}")
            if grad_error:
                print(f"  Error: {grad_error}")
                error_msg = f"{error_msg}; {grad_error}" if error_msg else grad_error
        else:
            print("  Skipped (check_backward=False)")
    else:
        print("\n  [Correctness] SKIPPED (--benchmark-only)")
        print("  [Gradient] SKIPPED (--benchmark-only)")

    # Benchmark
    print("\n  [Benchmark]")
    inputs = spec.input_factory(dtype=dtype, device=device)

    # Detach inputs for benchmark (no grad tracking)
    bench_inputs = tuple(
        x.detach() if isinstance(x, torch.Tensor) else x for x in inputs
    )

    # Profile reference
    print(f"  Profiling reference...")
    ref_result = benchmark_with_profiler(
        profiler,
        spec.reference_fn,
        bench_inputs,
        label=f"{spec.name}_ref",
    )
    ref_time = ref_result.time_ms

    # Profile kernel
    print(f"  Profiling kernel...")
    kernel_result = benchmark_with_profiler(
        profiler,
        kernel_fn,
        bench_inputs,
        label=f"{spec.name}_kernel",
    )
    kernel_time = kernel_result.time_ms

    speedup = ref_time / kernel_time if kernel_time > 0 else 0

    print(f"  Reference: {ref_time:.3f} ms")
    print(f"  Kernel:    {kernel_time:.3f} ms")
    print(f"  Speedup:   {speedup:.2f}x")

    # Print additional profiler metrics
    if kernel_result.metrics:
        print(f"  Metrics:")
        print_profiler_metrics(kernel_result.metrics, indent="    ")

    if kernel_result.output_path:
        print(f"  Profiler output: {kernel_result.output_path}")

    return ValidationResult(
        name=spec.name,
        correct=correct,
        grad_correct=grad_correct,
        ref_time_ms=ref_time,
        kernel_time_ms=kernel_time,
        speedup=speedup,
        error_msg=error_msg,
        max_abs_diff=max_abs,
        max_rel_diff=max_rel,
        profiler_name=profiler_name,
        profiler_metrics=kernel_result.metrics,
        profiler_output=str(kernel_result.output_path) if kernel_result.output_path else None,
    )


def save_result(result: ValidationResult):
    """Save validation result to JSON."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / f"{result.name}.json"
    data = {
        "name": result.name,
        "correct": result.correct,
        "grad_correct": result.grad_correct,
        "ref_time_ms": result.ref_time_ms,
        "kernel_time_ms": result.kernel_time_ms,
        "speedup": result.speedup,
        "error_msg": result.error_msg,
        "max_abs_diff": result.max_abs_diff,
        "max_rel_diff": result.max_rel_diff,
        "profiler_name": result.profiler_name,
        "profiler_metrics": result.profiler_metrics,
        "profiler_output": result.profiler_output,
    }
    path.write_text(json.dumps(data, indent=2) + "\n")
    print(f"\n  Result saved to: {path}")


def print_summary(results: list[ValidationResult]):
    """Print a summary table of all results."""
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"{'Kernel':<25} {'Correct':>8} {'Grad':>8} {'Speedup':>8} {'Profiler':>12} {'Status':>8}")
    print(f"{'-'*25} {'-'*8} {'-'*8} {'-'*8} {'-'*12} {'-'*8}")
    for r in results:
        correct_str = "PASS" if r.correct else "FAIL"
        grad_str = "PASS" if r.grad_correct else "FAIL"
        speedup_str = f"{r.speedup:.2f}x" if r.speedup > 0 else "N/A"
        profiler_str = r.profiler_name[:10]  # Truncate long names
        overall = "OK" if (r.correct and r.grad_correct) else "FAIL"
        print(f"{r.name:<25} {correct_str:>8} {grad_str:>8} {speedup_str:>8} {profiler_str:>12} {overall:>8}")

    # Print profiler summary if any non-wallclock profilers were used
    if any(r.profiler_name != "wallclock" for r in results):
        print(f"\nProfiler Outputs:")
        for r in results:
            if r.profiler_output:
                print(f"  {r.name}: {r.profiler_output}")


def main():
    parser = argparse.ArgumentParser(
        description="DSL Kernel Validation Harness",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("kernel", nargs="?", help="Kernel name to validate")
    parser.add_argument("--all", action="store_true", help="Validate all registered kernels")
    parser.add_argument("--benchmark-only", action="store_true", help="Skip correctness checks")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--list", action="store_true", help="List available kernel specs")
    parser.add_argument(
        "--list-profilers",
        action="store_true",
        help="List available profilers",
    )
    parser.add_argument(
        "--profiler", "-p",
        default=DEFAULT_PROFILER,
        choices=list_profilers(),
        help=f"Profiler to use for benchmarking (default: {DEFAULT_PROFILER})",
    )
    args = parser.parse_args()

    if args.list:
        specs = list_specs()
        print("Registered kernel specs:")
        for name in specs:
            print(f"  - {name}")
        return

    if args.list_profilers:
        print("Available profilers:")
        for name in list_profilers():
            profiler = get_profiler(name, output_dir=PROFILER_DIR / "_test")
            status = "available" if profiler.is_available() else "not available"
            print(f"  - {name}: {status}")
        return

    if not args.kernel and not args.all:
        parser.print_help()
        sys.exit(1)

    if args.all:
        names = list_specs()
    else:
        names = [args.kernel]

    results = []
    for name in names:
        spec = get_spec(name)
        result = validate_kernel(
            spec,
            profiler_name=args.profiler,
            verbose=args.verbose,
            benchmark_only=args.benchmark_only,
        )
        save_result(result)
        results.append(result)

    if len(results) > 1:
        print_summary(results)


if __name__ == "__main__":
    main()
