"""Pipeline orchestrator for AI-powered kernel generation.

This module provides the GenerationPipeline class that orchestrates
the generate -> validate -> refine loop for automated kernel development.

Usage:
    from experiments.generation.pipeline import GenerationPipeline, PipelineConfig

    config = PipelineConfig(
        provider="anthropic",
        model="claude-opus-4-20250514",
        backend="triton",
    )
    pipeline = GenerationPipeline(config)
    result = pipeline.run("rmsnorm")
"""

import hashlib
import json
import os
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

from experiments.generation.spec import KernelSpec, get_spec
from experiments.generation.harness import validate_kernel, ValidationResult
from experiments.generation.ai_providers import get_provider
from experiments.generation.prompts import build_prompt, build_refine_prompt


PROJECT_ROOT = Path(__file__).parent.parent.parent


@dataclass
class PipelineConfig:
    """Configuration for the generation pipeline.

    Attributes:
        provider: AI provider name (anthropic, openai, glm)
        model: Model identifier
        backend: Kernel backend (triton, tilelang)
        max_iterations: Maximum refinement iterations
        api_key_env: Environment variable name for API key
        cache_dir: Directory for caching results
        example_kernel_path: Optional path to example kernel for reference
        max_tokens: Maximum tokens for generation
        temperature: Temperature for generation (0.0 = deterministic)
        base_url: Optional custom base URL for API (e.g., for GLM OpenAI-compatible API)
    """
    provider: str = "anthropic"
    model: str = "claude-opus-4-20250514"
    backend: str = "triton"
    max_iterations: int = 3
    api_key_env: str = "ANTHROPIC_API_KEY"
    cache_dir: Path = field(default_factory=lambda: Path("experiments/generation/cache"))
    example_kernel_path: Optional[str] = None
    max_tokens: int = 8192
    temperature: float = 0.0
    base_url: Optional[str] = None

    def get_api_key(self) -> str:
        """Get API key from environment."""
        return os.environ.get(self.api_key_env, "")


@dataclass
class PipelineResult:
    """Result from running the generation pipeline.

    Attributes:
        spec_name: Name of the kernel spec
        success: Whether generation succeeded
        iterations: Number of iterations performed
        final_result: Final ValidationResult from harness
        code: Final generated code
        tokens_used: Total tokens consumed
        error_msg: Error message if failed
    """
    spec_name: str
    success: bool
    iterations: int
    final_result: ValidationResult
    code: str
    tokens_used: int
    error_msg: str = ""

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "spec_name": self.spec_name,
            "success": self.success,
            "iterations": self.iterations,
            "final_result": {
                "name": self.final_result.name,
                "correct": self.final_result.correct,
                "grad_correct": self.final_result.grad_correct,
                "ref_time_ms": self.final_result.ref_time_ms,
                "kernel_time_ms": self.final_result.kernel_time_ms,
                "speedup": self.final_result.speedup,
                "error_msg": self.final_result.error_msg,
                "max_abs_diff": self.final_result.max_abs_diff,
                "max_rel_diff": self.final_result.max_rel_diff,
            },
            "code": self.code,
            "tokens_used": self.tokens_used,
            "error_msg": self.error_msg,
        }


class GenerationPipeline:
    """Orchestrates AI kernel generation with validation loop.

    The pipeline:
    1. Generate initial kernel code
    2. Validate with harness
    3. If failed, refine with error feedback
    4. Repeat until success or max_iterations
    5. Cache results for reuse
    """

    def __init__(self, config: PipelineConfig):
        """Initialize the pipeline.

        Args:
            config: Pipeline configuration
        """
        self.config = config
        provider_kwargs = {
            "api_key": config.get_api_key(),
            "model": config.model,
        }
        if config.base_url:
            provider_kwargs["base_url"] = config.base_url
        self.provider = get_provider(config.provider, **provider_kwargs)
        self.cache_dir = config.cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def run(self, spec_name: str) -> PipelineResult:
        """Run the full generation pipeline.

        Args:
            spec_name: Name of the kernel spec to generate

        Returns:
            PipelineResult with generation outcome
        """
        spec = get_spec(spec_name)

        # Check cache
        cached = self._load_from_cache(spec_name)
        if cached and self._should_use_cache(cached):
            print(f"  [CACHE] Using cached result for {spec_name}")
            return cached

        # Build initial prompt
        prompt = self._build_prompt(spec)

        total_tokens = 0
        code = ""
        validation_result = None
        final_error_msg = ""

        for iteration in range(self.config.max_iterations):
            print(f"  [ITERATION {iteration + 1}/{self.config.max_iterations}]")

            # Generate code
            if iteration == 0:
                result = self.provider.generate_code(
                    prompt,
                    max_tokens=self.config.max_tokens,
                    temperature=self.config.temperature,
                )
            else:
                refine_prompt = self._build_refine_prompt(
                    spec,
                    validation_result,
                    code,
                    final_error_msg,
                )
                result = self.provider.generate_code(
                    refine_prompt,
                    max_tokens=self.config.max_tokens,
                    temperature=self.config.temperature,
                )

            code = result.code
            total_tokens += result.tokens_used

            print(f"    Tokens: {result.tokens_used}")
            print(f"    Finish reason: {result.finish_reason}")

            if result.finish_reason == "error":
                final_error_msg = f"API Error: {result.raw_response}"
                print(f"    ERROR: {final_error_msg}")
                break

            # Write kernel file
            self._write_kernel(spec, code)
            print(f"    Wrote: {spec.target_file}")

            # Validate
            validation_result = validate_kernel(
                spec,
                profiler_name="wallclock",
                verbose=False,
            )

            # Update error message
            if not validation_result.correct:
                final_error_msg = f"Correctness failed: {validation_result.error_msg}"
            elif not validation_result.grad_correct:
                final_error_msg = f"Gradient failed: {validation_result.error_msg}"
            else:
                final_error_msg = ""

            # Check success
            if validation_result.correct and validation_result.grad_correct:
                print(f"    SUCCESS: Speedup {validation_result.speedup:.2f}x")
                break
            else:
                print(f"    FAILED: {final_error_msg}")

        # Determine success
        success = (
            validation_result is not None
            and validation_result.correct
            and validation_result.grad_correct
        )

        # Save result
        final_result = PipelineResult(
            spec_name=spec_name,
            success=success,
            iterations=iteration + 1,
            final_result=validation_result or ValidationResult(
                name=spec_name,
                correct=False,
                grad_correct=False,
                ref_time_ms=0,
                kernel_time_ms=0,
                speedup=0,
                error_msg=final_error_msg,
            ),
            code=code,
            tokens_used=total_tokens,
            error_msg=final_error_msg,
        )
        self._save_to_cache(final_result)

        return final_result

    def _build_prompt(self, spec: KernelSpec) -> str:
        """Build initial generation prompt."""
        return build_prompt(
            spec,
            backend=self.config.backend,
            example_kernel_path=self.config.example_kernel_path,
        )

    def _build_refine_prompt(
        self,
        spec: KernelSpec,
        validation_result: ValidationResult,
        current_code: str,
        error_msg: str,
    ) -> str:
        """Build refinement prompt with error feedback."""
        return build_refine_prompt(
            spec,
            validation_result,
            current_code,
            error_msg,
        )

    def _write_kernel(self, spec: KernelSpec, code: str):
        """Write generated code to target file."""
        target = PROJECT_ROOT / spec.target_file
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(code)

    def _load_current_code(self, spec: KernelSpec) -> str:
        """Load current code from target file."""
        target = PROJECT_ROOT / spec.target_file
        if target.exists():
            return target.read_text()
        return "# No previous code"

    def _get_cache_key(self, spec_name: str) -> str:
        """Get cache key for a spec."""
        content = f"{spec_name}:{self.config.provider}:{self.config.model}:{self.config.backend}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def _load_from_cache(self, spec_name: str) -> Optional[PipelineResult]:
        """Load result from cache."""
        cache_file = self.cache_dir / f"{spec_name}.json"
        if cache_file.exists():
            try:
                data = json.loads(cache_file.read_text())
                # Reconstruct ValidationResult
                from dataclasses import fields

                final_result_data = data.pop("final_result", {})
                final_result = ValidationResult(**final_result_data)
                return PipelineResult(final_result=final_result, **data)
            except Exception as e:
                print(f"  [CACHE] Failed to load cache: {e}")
        return None

    def _save_to_cache(self, result: PipelineResult):
        """Save result to cache."""
        cache_file = self.cache_dir / f"{result.spec_name}.json"
        cache_file.write_text(json.dumps(result.to_dict(), indent=2) + "\n")

    def _should_use_cache(self, cached: PipelineResult) -> bool:
        """Determine if cached result should be used."""
        # Skip if cached result was a failure
        return cached.success
