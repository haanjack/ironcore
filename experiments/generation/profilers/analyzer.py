"""AI-based profiling analyzer for kernel optimization.

This module provides functionality to analyze profiling results using AI
and generate actionable optimization suggestions.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from experiments.generation.ai_providers import get_provider
from experiments.generation.spec import KernelSpec
from experiments.generation.harness import ValidationResult


@dataclass
class OptimizationAnalysis:
    """Result of AI-based profiling analysis.

    Attributes:
        bottleneck_identified: Description of the main bottleneck
        optimization_suggestions: List of specific optimization suggestions
        priority_rankings: List of (suggestion, priority_score) tuples
        estimated_impact: Estimated performance improvement potential
        code_areas_to_modify: List of specific code areas/regions to target
        raw_ai_response: Raw AI response for debugging
    """
    bottleneck_identified: str
    optimization_suggestions: list[str]
    priority_rankings: list[tuple[str, float]]  # (suggestion, priority 0-1)
    estimated_impact: str  # e.g., "10-20% improvement"
    code_areas_to_modify: list[str]
    raw_ai_response: str = ""

    def to_dict(self) -> dict:
        return {
            "bottle_identified": self.bottleneck_identified,
            "optimization_suggestions": self.optimization_suggestions,
            "priority_rankings": self.priority_rankings,
            "estimated_impact": self.estimated_impact,
            "code_areas_to_modify": self.code_areas_to_modify,
        }


class ProfilingAnalyzer:
    """AI-based analyzer for kernel profiling results.

    Analyzes profiling metrics and code to generate optimization strategies.
    """

    def __init__(self, provider_name: str = "openai", model: str = "glm-5",
                 api_key: str = None, base_url: str = None):
        """Initialize the analyzer.

        Args:
            provider_name: AI provider to use (openai, anthropic, etc.)
            model: Model name
            api_key: API key for the provider
            base_url: Optional custom base URL
        """
        provider_kwargs = {"api_key": api_key, "model": model}
        if base_url:
            provider_kwargs["base_url"] = base_url

        self.provider = get_provider(provider_name, **provider_kwargs)

    def analyze_results(
        self,
        spec: KernelSpec,
        validation_result: ValidationResult,
        current_code: str,
    ) -> OptimizationAnalysis:
        """Analyze profiling results and generate optimization suggestions.

        Args:
            spec: Kernel specification
            validation_result: Validation results including profiling metrics
            current_code: Current kernel implementation

        Returns:
            OptimizationAnalysis with suggestions
        """
        prompt = self._build_analysis_prompt(spec, validation_result, current_code)

        result = self.provider.generate_code(prompt, max_tokens=4096, temperature=0.3)

        response = result.raw_response or result.code

        return self._parse_analysis_response(response)

    def _build_analysis_prompt(
        self,
        spec: KernelSpec,
        validation_result: ValidationResult,
        current_code: str,
    ) -> str:
        """Build prompt for AI analysis.

        Args:
            spec: Kernel specification
            validation_result: Validation results
            current_code: Current kernel code

        Returns:
            Analysis prompt string
        """
        # Build metrics summary
        metrics_summary = self._format_metrics(validation_result)

        # Build performance gap analysis
        performance_gap = self._analyze_performance_gap(spec, validation_result)

        prompt = f"""You are a GPU kernel performance optimization expert. Analyze the following kernel implementation and profiling results to identify bottlenecks and suggest optimizations.

## Kernel Specification
Name: {spec.name}
Description: {spec.description}

## Current Performance
Reference Time: {validation_result.ref_time_ms:.3f} ms
Kernel Time: {validation_result.kernel_time_ms:.3f} ms
Speedup: {validation_result.speedup:.2f}x

{metrics_summary}

## Performance Gap
{performance_gap}

## Current Implementation
```python
{current_code[:5000]}  # Truncate if too long
```

## Optimization Hints
{chr(10).join(f'- {hint}' for hint in spec.optimization_hints) if spec.optimization_hints else 'None provided'}

## Your Task
Provide a structured analysis in the following JSON-like format:

```json
{{
  "bottleneck": "Main bottleneck description (e.g., memory bandwidth bound, poor occupancy, uncoalesced accesses)",
  "suggestions": [
    "Specific optimization suggestion 1",
    "Specific optimization suggestion 2",
    ...
  ],
  "priority": [
    ["suggestion_1", 0.9],
    ["suggestion_2", 0.7],
    ...
  ],
  "estimated_impact": "X-Y% improvement expected",
  "target_areas": [
    "Specific code region/line to modify",
    ...
  ]
}}
```

## Analysis Guidelines
1. Focus on Triton-specific optimizations (block size, tiling, vectorization, shared memory)
2. Consider memory access patterns (coalescing, stride, cache lines)
3. Evaluate compute vs memory intensity
4. Check for unnecessary synchronization or atomic operations
5. Suggest specific BLOCK_SIZE values and configurations
6. Recommend autotuner configurations if applicable

Return ONLY the JSON response, no additional text.
"""
        return prompt

    def _format_metrics(self, result: ValidationResult) -> str:
        """Format profiling metrics for analysis."""
        if not result.profiler_metrics:
            return "No detailed metrics available (wallclock timing only)."

        lines = ["Detailed Metrics:"]
        for key, value in sorted(result.profiler_metrics.items()):
            if isinstance(value, float):
                lines.append(f"  {key}: {value:.4f}")
            else:
                lines.append(f"  {key}: {value}")
        return "\n".join(lines)

    def _analyze_performance_gap(self, spec: KernelSpec, result: ValidationResult) -> str:
        """Analyze performance gap against targets."""
        met, unmet = spec.performance_targets.is_met(result)

        if met:
            return "✓ All performance targets met!"

        lines = ["Performance targets not met:"]
        for gap in unmet:
            lines.append(f"  - {gap}")

        # Calculate potential headroom
        if result.speedup > 0 and result.speedup < spec.performance_targets.min_speedup:
            headroom = (spec.performance_targets.min_speedup - result.speedup) / result.speedup * 100
            lines.append(f"\nAdditional {headroom:.1f}% speedup needed to meet target.")

        return "\n".join(lines)

    def _parse_analysis_response(self, response: str) -> OptimizationAnalysis:
        """Parse AI response into OptimizationAnalysis.

        Args:
            response: Raw AI response

        Returns:
            OptimizationAnalysis object
        """
        # Try to extract JSON from response
        import re

        json_match = re.search(r'```json\s*(\{.*?\})\s*```', response, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
        else:
            # Try to find JSON without code blocks
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
            else:
                # Fallback: parse as plain text
                return OptimizationAnalysis(
                    bottleneck_identified="Could not parse AI response",
                    optimization_suggestions=[],
                    priority_rankings=[],
                    estimated_impact="Unknown",
                    code_areas_to_modify=[],
                    raw_ai_response=response,
                )

        try:
            data = json.loads(json_str)
            return OptimizationAnalysis(
                bottleneck_identified=data.get("bottleneck", "Unknown"),
                optimization_suggestions=data.get("suggestions", []),
                priority_rankings=data.get("priority", []),
                estimated_impact=data.get("estimated_impact", "Unknown"),
                code_areas_to_modify=data.get("target_areas", []),
                raw_ai_response=response,
            )
        except json.JSONDecodeError:
            return OptimizationAnalysis(
                bottleneck_identified="Failed to parse AI JSON response",
                optimization_suggestions=[],
                priority_rankings=[],
                estimated_impact="Unknown",
                code_areas_to_modify=[],
                raw_ai_response=response,
            )


def get_analyzer(provider_name: str = "openai", model: str = None,
                api_key: str = None, base_url: str = None) -> ProfilingAnalyzer:
    """Get a profiling analyzer instance.

    Args:
        provider_name: AI provider name
        model: Model name (uses provider default if None)
        api_key: API key
        base_url: Optional custom base URL

    Returns:
        ProfilingAnalyzer instance
    """
    if model is None:
        # Use provider defaults
        model_defaults = {
            "openai": "gpt-4o",
            "glm": "glm-5",
            "kimi": "moonshot-v1-32k",
            "anthropic": "claude-opus-4-20250514",
        }
        model = model_defaults.get(provider_name, "gpt-4o")

    return ProfilingAnalyzer(
        provider_name=provider_name,
        model=model,
        api_key=api_key,
        base_url=base_url,
    )
