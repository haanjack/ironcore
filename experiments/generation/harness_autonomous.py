"""Autonomous AI-driven kernel generation and optimization harness.

This module provides a fully autonomous system that:
1. Generates initial kernel code using AI
2. Validates correctness
3. Runs detailed profiling
4. AI analyzes profiling results
5. AI creates optimization plan
6. AI generates optimized code
7. Loops until performance targets are met

Usage:
    python -m experiments.generation.harness_autonomous rmsnorm --provider glm
    python -m experiments.generation.harness_autonomous rmsnorm --provider glm --max-iterations 5
"""

import argparse
import importlib
import json
import os

# Load environment variables from .env file
from pathlib import Path
try:
    from dotenv import load_dotenv
    _env_path = Path(__file__).parent.parent.parent / ".env"
    if _env_path.exists():
        load_dotenv(_env_path)
except ImportError:
    pass  # dotenv not installed, will use environment variables directly
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

import torch

from experiments.generation.spec import KernelSpec, get_spec, list_specs
from experiments.generation.harness import (
    ValidationResult,
    _load_kernel_fn,
    check_correctness,
    check_gradient,
    benchmark_with_profiler,
    get_profiler,
    save_result,
)
from experiments.generation.profilers.analyzer import (
    ProfilingAnalyzer,
    OptimizationAnalysis,
    get_analyzer,
)
from experiments.generation.ai_providers import get_provider, resolve_provider_alias
from experiments.generation.prompts import (
    build_prompt,
    build_optimization_prompt,
    build_refine_prompt,
)


PROJECT_ROOT = Path(__file__).parent.parent.parent
RESULTS_DIR = Path(__file__).parent / "results"


@dataclass
class AutonomousIteration:
    """Result of a single autonomous iteration.

    Attributes:
        iteration_number: Iteration index (0-based)
        phase: Phase type ("generation", "correctness_fix", "optimization")
        validation_result: ValidationResult from this iteration
        analysis: Optional OptimizationAnalysis (for optimization phases)
        code_generated: Code generated in this iteration
        tokens_used: Tokens consumed
        timestamp: When this iteration completed
    """
    iteration_number: int
    phase: str  # "generation", "correctness_fix", "optimization"
    validation_result: ValidationResult
    analysis: Optional[OptimizationAnalysis] = None
    code_generated: str = ""
    tokens_used: int = 0
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> dict:
        return {
            "iteration": self.iteration_number,
            "phase": self.phase,
            "validation": {
                "correct": self.validation_result.correct,
                "grad_correct": self.validation_result.grad_correct,
                "speedup": self.validation_result.speedup,
                "kernel_time_ms": self.validation_result.kernel_time_ms,
                "error_msg": self.validation_result.error_msg,
            },
            "analysis": self.analysis.to_dict() if self.analysis else None,
            "tokens_used": self.tokens_used,
            "timestamp": self.timestamp,
        }


@dataclass
class AutonomousResult:
    """Final result from autonomous kernel generation.

    Attributes:
        spec_name: Name of the kernel spec
        success: Whether generation succeeded (correctness + performance targets met)
        iterations: List of all iterations performed
        total_tokens_used: Total tokens consumed
        final_code: Final generated kernel code
        total_time_seconds: Total time taken
        exit_reason: Reason for stopping (success, max_iterations, error)
    """
    spec_name: str
    success: bool
    iterations: list[AutonomousIteration]
    total_tokens_used: int
    final_code: str
    total_time_seconds: float
    exit_reason: str

    def to_dict(self) -> dict:
        return {
            "spec_name": self.spec_name,
            "success": self.success,
            "exit_reason": self.exit_reason,
            "iterations": [it.to_dict() for it in self.iterations],
            "total_tokens_used": self.total_tokens_used,
            "total_time_seconds": self.total_time_seconds,
        }

    def save(self, path: Path):
        """Save result to JSON file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2) + "\n")


class AutonomousHarness:
    """Autonomous AI-driven kernel generation and optimization system.

    The harness iterates through multiple phases:
    1. Initial code generation
    2. Correctness validation and fixes
    3. Performance profiling
    4. AI-based optimization analysis
    5. Optimized code generation
    6. Repeat until targets met or max iterations reached
    """

    def __init__(
        self,
        provider_name: str = "glm",
        model: str = None,
        api_key: str = None,
        base_url: str = None,
        max_iterations: int = 5,
        profiler_name: str = "wallclock",
        verbose: bool = True,
        timeout_seconds: int = 3600,
        auto_continue: bool = True,
    ):
        """Initialize the autonomous harness.

        Args:
            provider_name: AI provider name (glm, openai, anthropic, etc.)
            model: Model name (uses provider default if None)
            api_key: API key for the provider
            base_url: Optional custom base URL
            max_iterations: Maximum total iterations
            profiler_name: Profiler to use for benchmarking
            verbose: Enable verbose output
            timeout_seconds: Maximum time for entire generation (default: 1 hour)
            auto_continue: If True, continue iterations if making progress even after max_iterations
        """
        # Resolve provider alias and get configuration
        self.provider_alias, base_url_from_alias, default_model = resolve_provider_alias(provider_name)

        # Determine API key environment variable
        if self.provider_alias == "anthropic":
            api_key_env = "ANTHROPIC_API_KEY"
        else:
            api_key_env = "OPENAI_API_KEY"

        # Use provided values or defaults
        self.api_key = api_key or os.environ.get(api_key_env, "")
        self.base_url = base_url or base_url_from_alias
        self.model = model or default_model or ("gpt-4o" if self.provider_alias == "openai" else "glm-5")
        self.max_iterations = max_iterations
        self.profiler_name = profiler_name
        self.verbose = verbose
        self.timeout_seconds = timeout_seconds
        self.auto_continue = auto_continue

        # Initialize providers
        provider_kwargs = {"api_key": self.api_key, "model": self.model}
        if self.base_url:
            provider_kwargs["base_url"] = self.base_url

        self.code_provider = get_provider(self.provider_alias, **provider_kwargs)
        self.analyzer = ProfilingAnalyzer(
            provider_name=self.provider_alias,
            model=self.model,
            api_key=self.api_key,
            base_url=self.base_url,
        )

        self.start_time = None
        self.debug_dir = RESULTS_DIR / "debug" / datetime.now().strftime("%Y%m%d_%H%M%S")
        self.debug_dir.mkdir(parents=True, exist_ok=True)

    def run(self, spec: KernelSpec) -> AutonomousResult:
        """Run autonomous generation for a kernel spec.

        Args:
            spec: Kernel specification to generate

        Returns:
            AutonomousResult with full generation history
        """
        self.start_time = time.time()
        iterations = []
        total_tokens = 0

        current_code = ""
        phase = "generation"
        iteration_count = 0
        prev_error = None  # Track error for progress detection

        while True:
            # Check timeout
            elapsed = time.time() - self.start_time
            if elapsed > self.timeout_seconds:
                self._print_error(f"Timeout reached: {elapsed:.0f}s > {self.timeout_seconds}s")
                return self._finalize_result(spec, iterations, total_tokens, "timeout")

            # Check max iterations
            if iteration_count >= self.max_iterations:
                # Check if making progress and auto_continue is enabled
                if self.auto_continue and self._is_making_progress(iterations):
                    self._print_info(f"Auto-continuing: making progress (iteration {iteration_count})")
                else:
                    self._print_info(f"Max iterations ({self.max_iterations}) reached")
                    break

            self._print_header(f"Iteration {iteration_count + 1}: {phase} (elapsed: {elapsed:.0f}s)")

            # Determine what to do based on phase and previous results
            if iteration_count == 0:
                # Initial generation
                code, tokens = self._generate_initial_code(spec)
                current_code = code
                total_tokens += tokens
            elif iterations[-1].validation_result.correct and iterations[-1].validation_result.grad_correct:
                # Correctness passed, check if performance targets met
                targets_met, unmet = spec.performance_targets.is_met(iterations[-1].validation_result)

                if targets_met:
                    self._print_success("All targets met!")
                    return self._finalize_result(spec, iterations, total_tokens, "success")

                # Need to optimize
                self._print_info(f"Targets not met: {unmet}")

                # Run AI analysis
                analysis = self._analyze_performance(spec, iterations[-1].validation_result, current_code)
                total_tokens += len(analysis.raw_ai_response) // 4  # Rough estimate

                # Generate optimized code
                code, tokens = self._generate_optimized_code(
                    spec, iterations[-1].validation_result, analysis, current_code
                )
                current_code = code
                total_tokens += tokens
                phase = "optimization"

                # Create iteration record with analysis
                iteration = AutonomousIteration(
                    iteration_number=iteration_count,
                    phase=phase,
                    validation_result=iterations[-1].validation_result,  # Will be updated after validation
                    analysis=analysis,
                    code_generated=current_code,
                    tokens_used=tokens,
                )
            else:
                # Correctness failed - fix errors
                error_msg = iterations[-1].validation_result.error_msg
                self._print_error(f"Correctness failed: {error_msg}")

                code, tokens = self._fix_correctness(spec, iterations[-1].validation_result, current_code, error_msg, iteration_count)
                current_code = code
                total_tokens += tokens
                phase = "correctness_fix"

            # Write kernel and validate
            self._write_kernel(spec, current_code)
            validation_result = self._validate_kernel(spec)

            # Create iteration record
            iteration = AutonomousIteration(
                iteration_number=iteration_count,
                phase=phase,
                validation_result=validation_result,
                code_generated=current_code,
                tokens_used=tokens,
            )
            iterations.append(iteration)

            # Print summary
            self._print_iteration_summary(iteration)

            # Check if we should exit early
            if (validation_result.correct and validation_result.grad_correct and
                spec.performance_targets.is_met(validation_result)[0]):
                self._print_success("All targets met!")
                return self._finalize_result(spec, iterations, total_tokens, "success")

            iteration_count += 1

        # Max iterations reached
        final_result = iterations[-1].validation_result
        targets_met, _ = spec.performance_targets.is_met(final_result)

        # Provide recommendation
        self._print_info(f"Final status: correct={final_result.correct}, grad={final_result.grad_correct}")
        if not final_result.correct:
            self._print_info("Recommendation: Increase max_iterations or check error patterns")
        elif not final_result.grad_correct:
            self._print_info("Recommendation: Review backward pass implementation")
        elif not targets_met:
            self._print_info(f"Recommendation: Current speedup {final_result.speedup:.2f}x, target is {spec.performance_targets.min_speedup}x")

        return self._finalize_result(
            spec, iterations, total_tokens,
            "success" if targets_met and final_result.correct and final_result.grad_correct else "max_iterations"
        )

    def _is_making_progress(self, iterations: list) -> bool:
        """Check if iterations are making progress (errors decreasing)."""
        if len(iterations) < 2:
            return True

        # Compare last two iterations
        prev = iterations[-2].validation_result
        curr = iterations[-1].validation_result

        # If we just became correct, that's progress
        if not prev.correct and curr.correct:
            return True

        # If both are incorrect, check if error message changed (AI is trying different approaches)
        if not prev.correct and not curr.correct:
            # Different error means AI is trying different fixes
            return prev.error_msg != curr.error_msg

        # If correct, check if speedup improved
        if prev.correct and curr.correct:
            return curr.speedup > prev.speedup * 0.95  # At least not getting worse

        return False

    def _generate_initial_code(self, spec: KernelSpec) -> tuple[str, int]:
        """Generate initial kernel code.

        Returns:
            Tuple of (code, tokens_used)
        """
        self._print_info("Generating initial kernel code...")

        prompt = build_prompt(spec, backend="triton")

        # Use larger max_tokens for GLM which generates reasoning content
        result = self.code_provider.generate_code(prompt, max_tokens=16384, temperature=0.0)

        # Save debug info
        reasoning_content = ""
        if hasattr(result, 'raw_response'):
            # Try to parse reasoning_content from raw response
            import re
            reasoning_match = re.search(r"reasoning_content='([^']*)'", str(result.raw_response))
            if reasoning_match:
                reasoning_content = reasoning_match.group(1)

        # Save debug info
        debug_file = self.debug_dir / f"iter_0_generation_raw.txt"
        debug_content = f"=== PROMPT ===\n{prompt[:2000]}\n\n=== RAW RESPONSE ===\n{result.raw_response}\n\n=== REASONING CONTENT ===\n{reasoning_content[:5000] if reasoning_content else '(not extracted)'}\n\n=== EXTRACTED CODE ===\n{result.code}"
        debug_file.write_text(debug_content)

        self._print_info(f"Generated {len(result.code)} characters, {result.tokens_used} tokens")
        self._print_info(f"Finish reason: {result.finish_reason}")

        if len(result.code) == 0:
            self._print_error(f"No code extracted! Raw response saved to {debug_file}")

        return result.code, result.tokens_used

    def _fix_correctness(
        self, spec: KernelSpec, validation_result: ValidationResult,
        current_code: str, error_msg: str, iteration: int = 0
    ) -> tuple[str, int]:
        """Generate code to fix correctness issues.

        Returns:
            Tuple of (code, tokens_used)
        """
        self._print_info("Fixing correctness issues...")

        prompt = build_refine_prompt(spec, validation_result, current_code, error_msg)

        result = self.code_provider.generate_code(prompt, max_tokens=16384, temperature=0.0)

        # Save debug info
        debug_file = self.debug_dir / f"iter_{iteration}_fix_raw.txt"
        debug_file.write_text(f"=== ERROR ===\n{error_msg}\n\n=== RAW RESPONSE ===\n{result.raw_response}\n\n=== EXTRACTED CODE ===\n{result.code}")

        self._print_info(f"Generated {len(result.code)} characters, {result.tokens_used} tokens")

        if len(result.code) == 0:
            self._print_error(f"No code extracted! Raw response saved to {debug_file}")

        return result.code, result.tokens_used

    def _analyze_performance(
        self, spec: KernelSpec, validation_result: ValidationResult, current_code: str
    ) -> OptimizationAnalysis:
        """Analyze performance using AI.

        Returns:
            OptimizationAnalysis with suggestions
        """
        self._print_info("Analyzing performance with AI...")

        analysis = self.analyzer.analyze_results(spec, validation_result, current_code)

        self._print_info(f"Bottleneck: {analysis.bottleneck_identified}")
        self._print_info(f"Expected impact: {analysis.estimated_impact}")

        if self.verbose and analysis.optimization_suggestions:
            self._print_info("Top suggestions:")
            for suggestion, priority in sorted(analysis.priority_rankings, key=lambda x: -x[1])[:3]:
                self._print_info(f"  - [{priority:.0%}] {suggestion}")

        return analysis

    def _generate_optimized_code(
        self, spec: KernelSpec, validation_result: ValidationResult,
        analysis: OptimizationAnalysis, current_code: str
    ) -> tuple[str, int]:
        """Generate optimized code based on analysis.

        Returns:
            Tuple of (code, tokens_used)
        """
        self._print_info("Generating optimized kernel code...")

        prompt = build_optimization_prompt(spec, validation_result, analysis, current_code)

        result = self.code_provider.generate_code(prompt, max_tokens=16384, temperature=0.0)

        self._print_info(f"Generated {len(result.code)} characters, {result.tokens_used} tokens")

        return result.code, result.tokens_used

    def _validate_kernel(self, spec: KernelSpec) -> ValidationResult:
        """Validate current kernel implementation.

        Returns:
            ValidationResult
        """
        device = "cuda"
        dtype = torch.float32

        # Load kernel
        try:
            kernel_fn = _load_kernel_fn(spec)
        except (FileNotFoundError, AttributeError) as e:
            return ValidationResult(
                name=spec.name,
                correct=False,
                grad_correct=False,
                ref_time_ms=0,
                kernel_time_ms=0,
                speedup=0,
                error_msg=str(e),
                profiler_name=self.profiler_name,
            )

        # Check correctness
        correct, error_msg, max_abs, max_rel = check_correctness(spec, kernel_fn, dtype, device, self.verbose)

        # Check gradient
        grad_correct = True
        grad_error = ""
        if spec.check_backward:
            grad_correct, grad_error = check_gradient(spec, kernel_fn, dtype, device, self.verbose)
            if grad_error:
                error_msg = f"{error_msg}; {grad_error}" if error_msg else grad_error

        # Benchmark if correctness passed
        kernel_time = 0
        ref_time = 0
        speedup = 0
        profiler_metrics = {}

        if correct and grad_correct:
            profiler = get_profiler(self.profiler_name, output_dir=RESULTS_DIR / "profiler" / spec.name)

            inputs = spec.input_factory(dtype=dtype, device=device)
            bench_inputs = tuple(x.detach() if isinstance(x, torch.Tensor) else x for x in inputs)

            ref_result = benchmark_with_profiler(profiler, spec.reference_fn, bench_inputs, f"{spec.name}_ref")
            kernel_result = benchmark_with_profiler(profiler, kernel_fn, bench_inputs, f"{spec.name}_kernel")

            ref_time = ref_result.time_ms
            kernel_time = kernel_result.time_ms
            speedup = ref_time / kernel_time if kernel_time > 0 else 0
            profiler_metrics = kernel_result.metrics

            if self.verbose:
                self._print_info(f"Reference: {ref_time:.3f}ms, Kernel: {kernel_time:.3f}ms, Speedup: {speedup:.2f}x")

        return ValidationResult(
            name=spec.name,
            correct=correct,
            grad_correct=grad_correct,
            ref_time_ms=ref_time,
            kernel_time_ms=kernel_time,
            speedup=speedup,
            error_msg=error_msg,
            profiler_name=self.profiler_name,
            profiler_metrics=profiler_metrics,
        )

    def _write_kernel(self, spec: KernelSpec, code: str):
        """Write generated code to target file."""
        target = PROJECT_ROOT / spec.target_file
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(code)

    def _finalize_result(
        self, spec: KernelSpec, iterations: list[AutonomousIteration],
        total_tokens: int, exit_reason: str
    ) -> AutonomousResult:
        """Create final result object."""
        total_time = time.time() - self.start_time if self.start_time else 0

        final_result = iterations[-1].validation_result if iterations else None
        success = (
            exit_reason == "success" and
            final_result.correct and
            final_result.grad_correct
        )

        return AutonomousResult(
            spec_name=spec.name,
            success=success,
            iterations=iterations,
            total_tokens_used=total_tokens,
            final_code=iterations[-1].code_generated if iterations else "",
            total_time_seconds=total_time,
            exit_reason=exit_reason,
        )

    def _print_header(self, text: str):
        print(f"\n{'='*60}")
        print(f"{text}")
        print(f"{'='*60}")

    def _print_info(self, text: str):
        if self.verbose:
            print(f"  [INFO] {text}")

    def _print_error(self, text: str):
        print(f"  [ERROR] {text}")

    def _print_success(self, text: str):
        print(f"  [SUCCESS] {text}")

    def _print_iteration_summary(self, iteration: AutonomousIteration):
        """Print summary of iteration results."""
        vr = iteration.validation_result
        print(f"\n  Results:")
        print(f"    Correctness: {'PASS' if vr.correct else 'FAIL'}")
        print(f"    Gradient: {'PASS' if vr.grad_correct else 'FAIL'}")
        if vr.correct and vr.grad_correct:
            print(f"    Speedup: {vr.speedup:.2f}x")
        if vr.error_msg:
            print(f"    Error: {vr.error_msg}")


def main():
    parser = argparse.ArgumentParser(
        description="Autonomous AI-Driven Kernel Generation Harness",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("kernel", help="Kernel spec name to generate")
    parser.add_argument("--provider", "-p", default="glm", help="AI provider to use")
    parser.add_argument("--model", "-m", help="Model name (uses provider default if not specified)")
    parser.add_argument("--base-url", help="Custom API base URL")
    parser.add_argument("--max-iterations", type=int, default=5, help="Maximum iterations")
    parser.add_argument("--profiler", default="wallclock", help="Profiler to use")
    parser.add_argument("--api-key", help="API key (overrides environment)")
    parser.add_argument("--quiet", "-q", action="store_true", help="Reduce verbosity")
    parser.add_argument("--list", action="store_true", help="List available kernel specs")
    args = parser.parse_args()

    if args.list:
        print("Registered kernel specs:")
        for name in list_specs():
            print(f"  - {name}")
        return

    spec = get_spec(args.kernel)

    harness = AutonomousHarness(
        provider_name=args.provider,
        model=args.model,
        api_key=args.api_key,
        base_url=args.base_url,
        max_iterations=args.max_iterations,
        profiler_name=args.profiler,
        verbose=not args.quiet,
    )

    result = harness.run(spec)

    # Save result
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_path = RESULTS_DIR / f"autonomous_{args.kernel}_{timestamp}.json"
    result.save(result_path)

    # Print final summary
    print(f"\n{'='*60}")
    print("FINAL SUMMARY")
    print(f"{'='*60}")
    print(f"  Kernel: {result.spec_name}")
    print(f"  Status: {'SUCCESS' if result.success else 'FAILED'}")
    print(f"  Exit Reason: {result.exit_reason}")
    print(f"  Iterations: {len(result.iterations)}")
    print(f"  Total Tokens: {result.total_tokens_used}")
    print(f"  Total Time: {result.total_time_seconds:.1f}s")

    if result.iterations:
        final = result.iterations[-1].validation_result
        print(f"  Final Speedup: {final.speedup:.2f}x")

    print(f"  Result saved to: {result_path}")

    sys.exit(0 if result.success else 1)


if __name__ == "__main__":
    main()
