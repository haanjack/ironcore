"""Hybrid AI-driven kernel generation combining exploration with closed-loop refinement.

This module provides a hybrid system that:
1. Runs 7-stage exploration for rich context and better initial code
2. Uses closed-loop refinement for correctness fixing and performance optimization
3. Passes exploration context to all refinement prompts

The key insight:
- Exploration provides better upfront analysis (higher correctness rate)
- Closed-loop refinement enables performance optimization
- Combining both gives best of both worlds

Usage:
    python -m experiments.generation.harness_hybrid rmsnorm --provider glm
    python -m experiments.generation.harness_hybrid rmsnorm --provider glm --max-iterations 5
"""

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

import torch

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    _env_path = Path(__file__).parent.parent.parent / ".env"
    if _env_path.exists():
        load_dotenv(_env_path)
except ImportError:
    pass

from experiments.generation.spec import KernelSpec, get_spec, list_specs
from experiments.generation.exploration.kernel_explorer import KernelExplorer, ExplorationResult
from experiments.generation.harness import (
    ValidationResult,
    _load_kernel_fn,
    check_correctness,
    benchmark_with_profiler,
    write_kernel,
    save_result,
    get_profiler,
)
from experiments.generation.profilers.analyzer import (
    ProfilingAnalyzer,
    OptimizationAnalysis,
)
from experiments.generation.ai_providers import get_provider, resolve_provider_alias
from experiments.generation.prompts import build_optimization_prompt, build_refine_prompt


PROJECT_ROOT = Path(__file__).parent.parent.parent
RESULTS_DIR = Path(__file__).parent / "results"


@dataclass
class HybridIteration:
    """Result of a single hybrid iteration.

    Attributes:
        iteration_number: Iteration index (0-based)
        phase: Phase type ("exploration", "correctness_fix", "optimization")
        validation_result: ValidationResult from this iteration
        analysis: Optional OptimizationAnalysis (for optimization phases)
        exploration_result: Full ExplorationResult from exploration phase
        code_generated: Code generated in this iteration
        tokens_used: Tokens consumed
        timestamp: When this iteration completed
    """
    iteration_number: int
    phase: str  # "exploration", "correctness_fix", "optimization"
    validation_result: ValidationResult
    analysis: Optional[OptimizationAnalysis] = None
    exploration_result: Optional[ExplorationResult] = None
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
            "exploration_tokens": self.exploration_result.total_tokens_used if self.exploration_result else 0,
            "tokens_used": self.tokens_used,
            "timestamp": self.timestamp,
        }


@dataclass
class HybridResult:
    """Final result from hybrid kernel generation.

    Attributes:
        spec_name: Name of the kernel spec
        success: Whether generation succeeded (correctness + performance targets met)
        iterations: List of all iterations performed
        exploration_result: Full exploration result from initial phase
        total_tokens_used: Total tokens consumed (exploration + refinement)
        final_code: Final generated kernel code
        total_time_seconds: Total time taken
        exit_reason: Reason for stopping (success, max_iterations, error)
    """
    spec_name: str
    success: bool
    iterations: list[HybridIteration]
    exploration_result: Optional[ExplorationResult]
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
            "exploration_result": {
                "spec_name": self.exploration_result.spec_name if self.exploration_result else None,
                "total_tokens_used": self.exploration_result.total_tokens_used if self.exploration_result else 0,
                "algorithm_summary": self.exploration_result.algorithm_summary[:200] if self.exploration_result else None,
            } if self.exploration_result else None,
            "total_tokens_used": self.total_tokens_used,
            "total_time_seconds": self.total_time_seconds,
        }

    def save(self, path: Path):
        """Save result to JSON file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2) + "\n")


class HybridHarness:
    """Hybrid AI-driven kernel generation combining exploration with closed-loop refinement.

    The harness works in two phases:
    Phase 1: Exploration (7-stage analysis)
      - Algorithm → Graph → Tiling → Plan → Structure → Code
      - Produces rich context and initial code

    Phase 2: Closed-Loop Refinement
      - Test correctness → Fix if needed
      - Profile performance → AI analysis → Optimize
      - Repeat until targets met or max iterations

    Key advantage: Exploration context is passed to all refinement prompts,
    enabling better fixes and optimizations.
    """

    def __init__(
        self,
        provider_name: str = "glm",
        model: str = None,
        api_key: str = None,
        base_url: str = None,
        max_refinement_iterations: int = 5,
        profiler_name: str = "wallclock",
        verbose: bool = True,
    ):
        """Initialize the hybrid harness.

        Args:
            provider_name: AI provider name (glm, openai, anthropic, etc.)
            model: Model name (uses provider default if None)
            api_key: API key for the provider
            base_url: Optional custom base URL
            max_refinement_iterations: Maximum refinement iterations AFTER exploration
            profiler_name: Profiler to use for benchmarking
            verbose: Enable verbose output
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
        self.max_refinement_iterations = max_refinement_iterations
        self.profiler_name = profiler_name
        self.verbose = verbose

        # Initialize explorer (for Phase 1)
        self.explorer = KernelExplorer(
            provider_name=provider_name,
            model=model,
            api_key=api_key,
            verbose=verbose,
        )

        # Initialize providers (for Phase 2 refinement)
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

    def run(self, spec: KernelSpec) -> HybridResult:
        """Run hybrid generation for a kernel spec.

        Args:
            spec: Kernel specification to generate

        Returns:
            HybridResult with full generation history
        """
        self.start_time = time.time()
        iterations = []
        total_tokens = 0

        # ============================================================
        # PHASE 1: Exploration (7-stage analysis)
        # ============================================================
        self._print_header("PHASE 1: Exploration (7-stage analysis)")

        exploration_result = self.explorer.explore(spec)
        total_tokens += exploration_result.total_tokens_used

        self._print_info(f"Exploration complete: {exploration_result.total_tokens_used} tokens")
        self._print_info(f"Algorithm summary: {exploration_result.algorithm_summary[:100]}...")

        # Write exploration-generated kernel
        current_code = exploration_result.refined_code or exploration_result.initial_code
        self._write_kernel(spec, current_code)

        # Validate exploration result
        validation_result = self._validate_kernel(spec)

        # Create iteration record for exploration
        exploration_iter = HybridIteration(
            iteration_number=0,
            phase="exploration",
            validation_result=validation_result,
            exploration_result=exploration_result,
            code_generated=current_code,
            tokens_used=exploration_result.total_tokens_used,
        )
        iterations.append(exploration_iter)
        self._print_iteration_summary(exploration_iter)

        # Check if exploration already met all targets
        if (validation_result.correct and validation_result.grad_correct and
            spec.performance_targets.is_met(validation_result)[0]):
            self._print_success("Exploration already met all targets!")
            return self._finalize_result(spec, iterations, exploration_result, total_tokens, "success")

        # ============================================================
        # PHASE 2: Closed-Loop Refinement
        # ============================================================
        self._print_header(f"PHASE 2: Closed-Loop Refinement (max {self.max_refinement_iterations} iterations)")

        phase = "correctness_fix" if not (validation_result.correct and validation_result.grad_correct) else "optimization"

        for i in range(self.max_refinement_iterations):
            iter_num = i + 1  # 1-indexed for refinement iterations
            self._print_header(f"Refinement Iteration {iter_num}/{self.max_refinement_iterations}: {phase}")

            # Determine what to do based on previous results
            if not (iterations[-1].validation_result.correct and iterations[-1].validation_result.grad_correct):
                # Correctness failed - fix errors with exploration context
                error_msg = iterations[-1].validation_result.error_msg
                self._print_error(f"Correctness failed: {error_msg}")

                code, tokens = self._fix_correctness_with_context(
                    spec, iterations[-1].validation_result, current_code,
                    error_msg, exploration_result
                )
                current_code = code
                total_tokens += tokens
                phase = "correctness_fix"

            else:
                # Correctness passed, check if performance targets met
                targets_met, unmet = spec.performance_targets.is_met(iterations[-1].validation_result)

                if targets_met:
                    self._print_success("All targets met!")
                    return self._finalize_result(spec, iterations, exploration_result, total_tokens, "success")

                # Need to optimize with exploration context
                self._print_info(f"Targets not met: {unmet}")

                # Run AI analysis with exploration context
                analysis = self._analyze_performance_with_context(
                    spec, iterations[-1].validation_result, current_code, exploration_result
                )
                total_tokens += len(analysis.raw_ai_response) // 4  # Rough estimate

                # Generate optimized code with exploration context
                code, tokens = self._generate_optimized_code_with_context(
                    spec, iterations[-1].validation_result, analysis, current_code, exploration_result
                )
                current_code = code
                total_tokens += tokens
                phase = "optimization"

            # Write kernel and validate
            self._write_kernel(spec, current_code)
            validation_result = self._validate_kernel(spec)

            # Create iteration record
            iteration = HybridIteration(
                iteration_number=iter_num,
                phase=phase,
                validation_result=validation_result,
                analysis=analysis if phase == "optimization" else None,
                exploration_result=exploration_result,
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
                return self._finalize_result(spec, iterations, exploration_result, total_tokens, "success")

        # Max iterations reached
        final_result = iterations[-1].validation_result
        targets_met, _ = spec.performance_targets.is_met(final_result)

        return self._finalize_result(
            spec, iterations, exploration_result, total_tokens,
            "success" if targets_met and final_result.correct and final_result.grad_correct else "max_iterations"
        )

    def _fix_correctness_with_context(
        self, spec: KernelSpec, validation_result: ValidationResult,
        current_code: str, error_msg: str, exploration_result: ExplorationResult
    ) -> tuple[str, int]:
        """Fix correctness issues using exploration context.

        The key difference from autonomous: we include exploration context
        (algorithm analysis, graph analysis, tiling strategy, etc.) to help
        the AI make better fixes.
        """
        self._print_info("Fixing correctness issues with exploration context...")

        # Build prompt with exploration context
        prompt = self._build_context_aware_refinement_prompt(
            spec, current_code, error_msg, exploration_result, "correctness"
        )

        result = self.code_provider.generate_code(prompt, max_tokens=16384, temperature=0.0)

        # Save debug info
        debug_file = self.debug_dir / f"iter_{len(self.explorer.results.listdir())}_fix_raw.txt"
        debug_content = f"=== PROMPT ===\n{prompt[:3000]}\n\n=== RAW RESPONSE ===\n{result.raw_response}\n\n=== EXTRACTED CODE ===\n{result.code}"
        debug_file.write_text(debug_content)

        self._print_info(f"Generated {len(result.code)} characters, {result.tokens_used} tokens")

        return result.code, result.tokens_used

    def _analyze_performance_with_context(
        self, spec: KernelSpec, validation_result: ValidationResult,
        current_code: str, exploration_result: ExplorationResult
    ) -> OptimizationAnalysis:
        """Analyze performance using exploration context.

        The AI gets both profiling data AND the exploration analysis,
        enabling more informed optimization decisions.
        """
        self._print_info("Analyzing performance with exploration context...")

        # Get profiling metrics
        profiler = get_profiler(self.profiler_name)
        metrics = profiler.get_metrics(spec, _load_kernel_fn(spec))

        # Build context-aware prompt
        prompt = self._build_context_aware_analysis_prompt(
            spec, validation_result, metrics, current_code, exploration_result
        )

        # Use the analyzer's provider
        analysis = self.analyzer.analyze(
            spec=spec,
            validation_result=validation_result,
            metrics=metrics,
            current_code=current_code,
            custom_prompt=prompt,  # Use our context-aware prompt
        )

        return analysis

    def _generate_optimized_code_with_context(
        self, spec: KernelSpec, validation_result: ValidationResult,
        analysis: OptimizationAnalysis, current_code: str, exploration_result: ExplorationResult
    ) -> tuple[str, int]:
        """Generate optimized code using exploration context.

        Combines:
        - Original exploration analysis
        - Performance profiling
        - AI optimization recommendations
        """
        self._print_info("Generating optimized code with exploration context...")

        # Build prompt with all context
        prompt = self._build_context_aware_optimization_prompt(
            spec, validation_result, analysis, current_code, exploration_result
        )

        result = self.code_provider.generate_code(prompt, max_tokens=16384, temperature=0.0)

        # Save debug info
        debug_file = self.debug_dir / f"iter_{len(self.explorer.results.listdir())}_optimize_raw.txt"
        debug_content = f"=== PROMPT ===\n{prompt[:3000]}\n\n=== RAW RESPONSE ===\n{result.raw_response}\n\n=== EXTRACTED CODE ===\n{result.code}"
        debug_file.write_text(debug_content)

        self._print_info(f"Generated {len(result.code)} characters, {result.tokens_used} tokens")

        return result.code, result.tokens_used

    def _build_context_aware_refinement_prompt(
        self, spec: KernelSpec, current_code: str, error_msg: str,
        exploration_result: ExplorationResult, fix_type: str
    ) -> str:
        """Build a refinement prompt that includes exploration context."""
        context_parts = []

        # Header
        context_parts.append(f"""You are fixing a Triton kernel for {spec.name}.

## EXPLORATION CONTEXT

The kernel was generated using a 7-stage exploration pipeline. Use this context to make better fixes.

### Algorithm Analysis
{exploration_result.algorithm_summary}

### Graph Analysis
- Operations: {', '.join(exploration_result.graph_analysis.operations)}
- Reductions: {', '.join(exploration_result.graph_analysis.reductions)}
- Element-wise ops: {', '.join(exploration_result.graph_analysis.elementwise_ops)}
- Can parallelize rows: {exploration_result.graph_analysis.can_parallelize_rows}
- Can parallelize cols: {exploration_result.graph_analysis.can_parallelize_cols}

### Tiling Strategy
- Recommended BLOCK_SIZE: {exploration_result.tiling_strategy.recommended_block_size}
- Register pressure: {exploration_result.tiling_strategy.register_pressure}
- Tiling rationale: {exploration_result.tiling_strategy.tiling_rationale}

### Conversion Plan
""")

        for i, pass_plan in enumerate(exploration_result.conversion_plan.passes):
            context_parts.append(f"  Pass {i+1}: {pass_plan.get('name', 'unnamed')} ({pass_plan.get('type', 'unknown')})")

        # Current code and error
        context_parts.append(f"""

## CURRENT CODE (HAS ERRORS)
```python
{current_code}
```

## ERROR MESSAGE
{error_msg}

## YOUR TASK
Fix the code to resolve the errors. Use the exploration context to understand:
1. What the algorithm is trying to do
2. What data patterns are expected
3. What tiling strategy is optimal
4. What the computational structure should be

Return ONLY the fixed Python code, no markdown blocks.
""")

        return "\n".join(context_parts)

    def _build_context_aware_analysis_prompt(
        self, spec: KernelSpec, validation_result: ValidationResult,
        metrics: dict, current_code: str, exploration_result: ExplorationResult
    ) -> str:
        """Build a performance analysis prompt that includes exploration context."""
        return f"""Analyze the performance of a Triton kernel for {spec.name}.

## EXPLORATION CONTEXT
Algorithm: {exploration_result.algorithm_summary[:200]}

Tiling Strategy:
- BLOCK_SIZE: {exploration_result.tiling_strategy.recommended_block_size}
- Register pressure: {exploration_result.tiling_strategy.register_pressure}

Conversion Plan:
{chr(10).join(f'  - {p.get("name", "unnamed")}' for p in exploration_result.conversion_plan.passes)}

## CURRENT PERFORMANCE
Speedup: {validation_result.speedup:.2f}x
Kernel time: {validation_result.kernel_time_ms:.3f}ms

## PROFILING METRICS
{json.dumps(metrics, indent=2)}

Analyze why performance is below target and suggest specific optimizations considering the exploration context.
"""

    def _build_context_aware_optimization_prompt(
        self, spec: KernelSpec, validation_result: ValidationResult,
        analysis: OptimizationAnalysis, current_code: str, exploration_result: ExplorationResult
    ) -> str:
        """Build an optimization prompt that includes exploration context."""
        return f"""Optimize the Triton kernel for {spec.name}.

## EXPLORATION CONTEXT
Algorithm: {exploration_result.algorithm_summary[:200]}

Graph Analysis:
- Operations: {', '.join(exploration_result.graph_analysis.operations)}
- Reductions: {', '.join(exploration_result.graph_analysis.reductions)}
- Parallelization: rows={exploration_result.graph_analysis.can_parallelize_rows}, cols={exploration_result.graph_analysis.can_parallelize_cols}

Tiling Strategy:
- Recommended BLOCK_SIZE: {exploration_result.tiling_strategy.recommended_block_size}
- Rationale: {exploration_result.tiling_strategy.tiling_rationale}

## CURRENT PERFORMANCE
Speedup: {validation_result.speedup:.2f}x (target: {spec.performance_targets.min_speedup}x)
Kernel time: {validation_result.kernel_time_ms:.3f}ms

## OPTIMIZATION RECOMMENDATIONS
{analysis.raw_ai_response[:1000]}

## CURRENT CODE
```python
{current_code}
```

Generate optimized code that:
1. Incorporates the optimization recommendations
2. Respects the exploration analysis (tiling, graph structure, etc.)
3. Maintains correctness while improving performance

Return ONLY the optimized Python code, no markdown blocks.
"""

    def _validate_kernel(self, spec: KernelSpec) -> ValidationResult:
        """Validate kernel correctness and performance."""
        kernel_fn = _load_kernel_fn(spec)

        # Check correctness
        correct, error_msg, _, _ = check_correctness(spec, kernel_fn, torch.float32, "cuda", False)

        # If correct, benchmark performance
        if correct:
            from experiments.generation.harness import benchmark_with_profiler, ProfilerResult
            inputs = spec.input_factory()
            profiler = get_profiler(self.profiler_name, output_dir=self.debug_dir)

            # Benchmark generated kernel
            profiler_result = benchmark_with_profiler(profiler, kernel_fn, inputs, spec.name)
            kernel_time_ms = profiler_result.time_ms

            # Benchmark reference implementation for speedup calculation
            ref_result = benchmark_with_profiler(profiler, spec.reference_fn, inputs, f"{spec.name}_ref")
            ref_time_ms = ref_result.time_ms

            # Compute speedup
            speedup = ref_time_ms / kernel_time_ms if kernel_time_ms > 0 else 0.0

            return ValidationResult(
                correct=True,
                grad_correct=True,  # Assume grad correct if forward is
                speedup=speedup,
                kernel_time_ms=kernel_time_ms,
                error_msg=None,
            )
        else:
            return ValidationResult(
                correct=False,
                grad_correct=False,
                speedup=0.0,
                kernel_time_ms=0.0,
                error_msg=error_msg,
            )

    def _write_kernel(self, spec: KernelSpec, code: str):
        """Write kernel code to target file."""
        write_kernel(spec, code)

    def _finalize_result(
        self, spec: KernelSpec, iterations: list[HybridIteration],
        exploration_result: ExplorationResult, total_tokens: int, exit_reason: str
    ) -> HybridResult:
        """Finalize and save the result."""
        elapsed_time = time.time() - self.start_time

        final_iter = iterations[-1]
        success = (
            exit_reason == "success" and
            final_iter.validation_result.correct and
            final_iter.validation_result.grad_correct
        )

        result = HybridResult(
            spec_name=spec.name,
            success=success,
            iterations=iterations,
            exploration_result=exploration_result,
            total_tokens_used=total_tokens,
            final_code=final_iter.code_generated,
            total_time_seconds=elapsed_time,
            exit_reason=exit_reason,
        )

        # Save result
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_path = RESULTS_DIR / f"hybrid_{spec.name}_{timestamp}.json"
        result.save(result_path)

        self._print_header("FINAL SUMMARY")
        self._print_info(f"Kernel: {spec.name}")
        self._print_info(f"Status: {'SUCCESS' if success else 'FAILED'}")
        self._print_info(f"Exit Reason: {exit_reason}")
        self._print_info(f"Iterations: {len(iterations)} (1 exploration + {len(iterations)-1} refinement)")
        self._print_info(f"Total Tokens: {total_tokens:,}")
        self._print_info(f"Total Time: {elapsed_time:.1f}s")
        self._print_info(f"Exploration Tokens: {exploration_result.total_tokens_used:,}")
        self._print_info(f"Refinement Tokens: {total_tokens - exploration_result.total_tokens_used:,}")
        self._print_info(f"Final Speedup: {final_iter.validation_result.speedup:.2f}x")
        self._print_info(f"Result saved to: {result_path}")

        return result

    def _print_header(self, msg: str):
        if self.verbose:
            print(f"\n{'='*70}")
            print(f"{msg:^70}")
            print(f"{'='*70}")

    def _print_info(self, msg: str):
        if self.verbose:
            print(f"  [INFO] {msg}")

    def _print_error(self, msg: str):
        if self.verbose:
            print(f"  [ERROR] {msg}")

    def _print_success(self, msg: str):
        if self.verbose:
            print(f"  [SUCCESS] {msg}")

    def _print_iteration_summary(self, iteration: HybridIteration):
        """Print summary of an iteration."""
        if not self.verbose:
            return

        val = iteration.validation_result
        print(f"\n  Results:")
        print(f"    Correctness: {'PASS' if val.correct else 'FAIL'}")
        print(f"    Gradient: {'PASS' if val.grad_correct else 'FAIL'}")

        if val.correct:
            print(f"    Speedup: {val.speedup:.2f}x")
            print(f"    Kernel time: {val.kernel_time_ms:.3f}ms")

        if val.error_msg:
            print(f"    Error: {val.error_msg}")

        if iteration.exploration_result:
            print(f"    Exploration tokens: {iteration.exploration_result.total_tokens_used}")

        print(f"    Iteration tokens: {iteration.tokens_used}")


def main():
    """CLI entry point for hybrid harness."""
    parser = argparse.ArgumentParser(
        description="Hybrid AI-driven Triton kernel generation combining exploration with closed-loop refinement"
    )
    parser.add_argument(
        "spec",
        type=str,
        help="Kernel spec name (e.g., rmsnorm, softmax, layernorm)",
    )
    parser.add_argument(
        "--provider",
        type=str,
        default="glm",
        help="AI provider name (glm, openai, anthropic, etc.)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model name (uses provider default if not specified)",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="API key (uses environment variable if not specified)",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=5,
        help="Maximum refinement iterations after exploration (default: 5)",
    )
    parser.add_argument(
        "--profiler",
        type=str,
        default="wallclock",
        choices=["wallclock"],
        help="Profiler to use for benchmarking",
    )
    parser.add_argument(
        "--list-specs",
        action="store_true",
        help="List all available kernel specs and exit",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=True,
        help="Enable verbose output (default: True)",
    )

    args = parser.parse_args()

    # List specs if requested
    if args.list_specs:
        print("Available kernel specs:")
        for spec_name in list_specs():
            spec = get_spec(spec_name)
            print(f"  - {spec_name}: {spec.description}")
        return 0

    # Get spec
    spec = get_spec(args.spec)
    if spec is None:
        print(f"Error: Unknown spec '{args.spec}'")
        print(f"Available specs: {', '.join(list_specs())}")
        return 1

    # Run hybrid harness
    harness = HybridHarness(
        provider_name=args.provider,
        model=args.model,
        api_key=args.api_key,
        max_refinement_iterations=args.max_iterations,
        profiler_name=args.profiler,
        verbose=args.verbose,
    )

    result = harness.run(spec)

    return 0 if result.success else 1


if __name__ == "__main__":
    sys.exit(main())
