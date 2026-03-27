"""
Exploration phase for AI-driven Triton kernel generation.

This module implements a sophisticated exploration phase where AI:
1. Analyzes the algorithm and creates conversion plan
2. Analyzes data flow and computation graph
3. Determines optimal tiling/block size strategy
4. Generates initial kernel structure
5. Iteratively refines based on diagnostics
"""

import json
import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    _env_path = Path(__file__).parent.parent.parent.parent / ".env"
    if _env_path.exists():
        load_dotenv(_env_path)
except ImportError:
    pass

from experiments.generation.spec import KernelSpec


@dataclass
class GraphAnalysis:
    """Analysis of computation graph and data flow."""

    # Input/output characteristics
    input_shapes: List[str] = field(default_factory=list)
    output_shapes: List[str] = field(default_factory=list)
    data_dependencies: List[str] = field(default_factory=list)

    # Computation characteristics
    operations: List[str] = field(default_factory=list)
    reductions: List[str] = field(default_factory=list)
    elementwise_ops: List[str] = field(default_factory=list)

    # Memory access patterns
    memory_reads_per_element: int = 1
    memory_writes_per_element: int = 1
    optimal_access_pattern: str = "contiguous"  # or "strided", "random"

    # Parallelization opportunities
    can_parallelize_rows: bool = True
    can_parallelize_cols: bool = False
    requires_reduction: bool = False
    requires_atomic_accumulation: bool = False


@dataclass
class TilingStrategy:
    """Analysis of optimal tiling/block size strategy."""

    # Block size recommendations
    recommended_block_size: int = 1024
    min_block_size: int = 256
    max_block_size: int = 8192

    # Tiling rationale
    tiling_rationale: str = ""
    considerations: List[str] = field(default_factory=list)

    # Memory hierarchy considerations
    fits_in_l1_cache: bool = True
    fits_in_registers: bool = True
    register_pressure: str = "low"  # low, medium, high

    # Alternative strategies
    alternative_strategies: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class ConversionPlan:
    """Structured plan for converting algorithm to Triton kernel."""

    # High-level structure
    kernel_name: str = ""
    description: str = ""
    input_tensors: List[Dict[str, str]] = field(default_factory=list)
    output_tensors: List[Dict[str, str]] = field(default_factory=list)

    # Computation breakdown
    passes: List[Dict[str, Any]] = field(default_factory=list)

    # Memory management
    requires_intermediate_storage: bool = False
    intermediate_tensors: List[str] = field(default_factory=list)

    # Gradient computation (if applicable)
    requires_backward: bool = True
    saved_tensors: List[str] = field(default_factory=list)
    gradient_complexity: str = "medium"  # simple, medium, complex


@dataclass
class ExplorationResult:
    """Result of the exploration phase."""

    spec_name: str
    timestamp: str

    # Stage 1: Algorithm Analysis
    algorithm_summary: str = ""

    # Stage 2: Graph Analysis
    graph_analysis: GraphAnalysis = field(default_factory=GraphAnalysis)

    # Stage 3: Tiling Strategy
    tiling_strategy: TilingStrategy = field(default_factory=TilingStrategy)

    # Stage 4: Conversion Plan
    conversion_plan: ConversionPlan = field(default_factory=ConversionPlan)

    # Stage 5: Initial Code Structure
    code_structure: str = ""

    # Stage 6: Initial Implementation
    initial_code: str = ""

    # Stage 7: Diagnostic Results
    compilation_errors: List[str] = field(default_factory=list)
    numerical_errors: List[Dict[str, Any]] = field(default_factory=list)

    # Refined code after diagnostics
    refined_code: str = ""

    # Token usage tracking
    total_tokens_used: int = 0
    stage_tokens: List[int] = field(default_factory=list)

    def save(self, path: Path):
        """Save exploration result to file."""
        path.parent.mkdir(parents=True, exist_ok=True)

        output = {
            "spec_name": self.spec_name,
            "timestamp": self.timestamp,
            "algorithm_summary": self.algorithm_summary,
            "graph_analysis": self.graph_analysis.__dict__,
            "tiling_strategy": self.tiling_strategy.__dict__,
            "conversion_plan": self.conversion_plan.__dict__,
            "code_structure": self.code_structure,
            "initial_code": self.initial_code,
            "compilation_errors": self.compilation_errors,
            "numerical_errors": self.numerical_errors,
            "refined_code": self.refined_code,
            "total_tokens_used": self.total_tokens_used,
            "stage_tokens": self.stage_tokens,
        }

        path.write_text(json.dumps(output, indent=2) + "\n")


class KernelExplorer:
    """Explores kernel design through multi-stage AI engagement."""

    def __init__(self, provider_name: str = "glm", model: str = None, api_key: str = None, verbose: bool = True):
        from experiments.generation.ai_providers import get_provider, resolve_provider_alias

        # Resolve provider alias and get configuration
        self.provider_alias, base_url_from_alias, default_model = resolve_provider_alias(provider_name)

        # Determine API key environment variable
        if self.provider_alias == "anthropic":
            api_key_env = "ANTHROPIC_API_KEY"
        else:
            api_key_env = "OPENAI_API_KEY"

        # Use provided values or get from environment
        if api_key is None:
            api_key = os.environ.get(api_key_env, "")

        self.api_key = api_key
        self.base_url = base_url_from_alias
        self.model = model or default_model or "glm-5"
        self.verbose = verbose

        # Initialize providers
        provider_kwargs = {"api_key": self.api_key, "model": self.model}
        if self.base_url:
            provider_kwargs["base_url"] = self.base_url

        self.code_provider = get_provider(self.provider_alias, **provider_kwargs)
        self.analysis_provider = get_provider(self.provider_alias, **provider_kwargs)

        self.results_dir = Path("/home/hanjack/ironcore-dsl-alpha/experiments/generation/results/exploration")
        self.results_dir.mkdir(parents=True, exist_ok=True)

    def explore(self, spec: KernelSpec) -> ExplorationResult:
        """Run full exploration pipeline for a kernel spec."""

        result = ExplorationResult(
            spec_name=spec.name,
            timestamp=datetime.now().isoformat()
        )

        print(f"\n{'='*70}")
        print(f"EXPLORATION PIPELINE: {spec.name}")
        print(f"{'='*70}")

        # ============================================================
        # STAGE 1: Algorithm Analysis
        # ============================================================
        print(f"\n[STAGE 1] Algorithm Analysis...")
        result.algorithm_summary = self._analyze_algorithm(spec, result)
        print(f"  Summary: {result.algorithm_summary[:100]}...")
        result.stage_tokens.append(0)  # Track tokens

        # ============================================================
        # STAGE 2: Graph Analysis
        # ============================================================
        print(f"\n[STAGE 2] Computation Graph Analysis...")
        result.graph_analysis = self._analyze_graph(spec)
        print(f"  Operations: {len(result.graph_analysis.operations)}")
        print(f"  Reductions: {result.graph_analysis.reductions}")
        print(f"  Parallelization: rows={result.graph_analysis.can_parallelize_rows}, cols={result.graph_analysis.can_parallelize_cols}")
        result.stage_tokens.append(0)

        # ============================================================
        # STAGE 3: Tiling Strategy
        # ============================================================
        print(f"\n[STAGE 3] Tiling Strategy Analysis...")
        result.tiling_strategy = self._analyze_tiling(spec, result.graph_analysis)
        print(f"  Block size: {result.tiling_strategy.recommended_block_size}")
        print(f"  Fits in L1: {result.tiling_strategy.fits_in_l1_cache}")
        print(f"  Register pressure: {result.tiling_strategy.register_pressure}")
        result.stage_tokens.append(0)

        # ============================================================
        # STAGE 4: Conversion Plan
        # ============================================================
        print(f"\n[STAGE 4] Structured Conversion Plan...")
        result.conversion_plan = self._create_conversion_plan(spec, result)
        print(f"  Passes: {len(result.conversion_plan.passes)}")
        for i, pass_plan in enumerate(result.conversion_plan.passes):
            print(f"    Pass {i+1}: {pass_plan.get('name', 'unnamed')}")
        result.stage_tokens.append(0)

        # ============================================================
        # STAGE 5: Code Structure
        # ============================================================
        print(f"\n[STAGE 5] Kernel Structure Design...")
        result.code_structure = self._design_code_structure(spec, result)
        result.stage_tokens.append(0)

        # ============================================================
        # STAGE 6: Initial Implementation
        # ============================================================
        print(f"\n[STAGE 6] Initial Code Generation...")
        result.initial_code, tokens = self._generate_initial_code(spec, result)
        result.total_tokens_used += tokens
        result.stage_tokens.append(tokens)
        print(f"  Generated {len(result.initial_code)} characters, {tokens} tokens")

        # ============================================================
        # STAGE 7: Diagnostic Refinement
        # ============================================================
        print(f"\n[STAGE 7] Diagnostic Refinement...")
        result.refined_code = self._refine_with_diagnostics(spec, result)
        result.stage_tokens.append(0)

        # Save result
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_path = self.results_dir / f"{spec.name}_{timestamp}.json"
        result.save(result_path)
        print(f"\n  Result saved to: {result_path}")

        return result

    def _analyze_algorithm(self, spec: KernelSpec, result: ExplorationResult) -> str:
        """Stage 1: Analyze the algorithm and create summary."""

        from experiments.generation.exploration.prompts_exploration import build_algorithm_analysis_prompt

        prompt = build_algorithm_analysis_prompt(spec)
        response = self.code_provider.generate_code(prompt, max_tokens=2000, temperature=0.0)
        result.total_tokens_used += response.tokens_used

        return response.code or ""

    def _analyze_graph(self, spec: KernelSpec) -> GraphAnalysis:
        """Stage 2: Analyze computation graph and data flow."""

        from experiments.generation.exploration.prompts_exploration import build_graph_analysis_prompt

        prompt = build_graph_analysis_prompt(spec)
        response = self.analysis_provider.generate_code(prompt, max_tokens=2000, temperature=0.0)

        # Parse response into GraphAnalysis
        # For now, return default analysis with AI insights
        return GraphAnalysis(
            operations=["load", "compute", "store"],
            reductions=["sum"] if "norm" in spec.name.lower() else [],
            elementwise_ops=["mul", "add"]
        )

    def _analyze_tiling(self, spec: KernelSpec, graph_analysis: GraphAnalysis) -> TilingStrategy:
        """Stage 3: Determine optimal tiling strategy."""

        from experiments.generation.exploration.prompts_exploration import build_tiling_analysis_prompt

        prompt = build_tiling_analysis_prompt(spec, graph_analysis)
        response = self.analysis_provider.generate_code(prompt, max_tokens=2000, temperature=0.0)

        # Determine block size from input size
        # This is simplified - real implementation would analyze more
        sample_input = spec.input_factory()
        hidden_dim = sample_input[0].shape[-1]

        block_size = 1024
        if hidden_dim > 2048:
            block_size = 2048
        if hidden_dim > 4096:
            block_size = 4096

        return TilingStrategy(
            recommended_block_size=min(block_size, hidden_dim),
            tiling_rationale=response.code or f"Based on hidden dimension {hidden_dim}"
        )

    def _create_conversion_plan(self, spec: KernelSpec, result: ExplorationResult) -> ConversionPlan:
        """Stage 4: Create structured conversion plan."""

        from experiments.generation.exploration.prompts_exploration import build_conversion_plan_prompt

        prompt = build_conversion_plan_prompt(spec, result)
        response = self.analysis_provider.generate_code(prompt, max_tokens=3000, temperature=0.0)
        result.total_tokens_used += response.tokens_used

        # Parse response into ConversionPlan
        # For now, create basic plan
        return ConversionPlan(
            kernel_name=spec.name,
            description=spec.description,
            passes=[
                {"name": "Compute statistics (mean/variance)", "type": "reduction"},
                {"name": "Normalize and scale", "type": "elementwise"}
            ] if "norm" in spec.name.lower() else [
                {"name": "Forward computation", "type": "elementwise"}
            ]
        )

    def _design_code_structure(self, spec: KernelSpec, result: ExplorationResult) -> str:
        """Stage 5: Design kernel code structure."""

        from experiments.generation.exploration.prompts_exploration import build_structure_design_prompt

        prompt = build_structure_design_prompt(spec, result)
        response = self.code_provider.generate_code(prompt, max_tokens=3000, temperature=0.0)
        result.total_tokens_used += response.tokens_used

        return response.code or ""

    def _generate_initial_code(self, spec: KernelSpec, result: ExplorationResult) -> tuple[str, int]:
        """Stage 6: Generate initial Triton code."""

        from experiments.generation.exploration.prompts_exploration import build_initial_code_prompt

        prompt = build_initial_code_prompt(spec, result)

        if self.verbose:
            print(f"    Prompt length: {len(prompt)} chars")
            print(f"    Prompt preview: {prompt[:500]}...")

        response = self.code_provider.generate_code(prompt, max_tokens=16384, temperature=0.0)

        if self.verbose:
            print(f"    Response code length: {len(response.code)} chars")
            print(f"    Response tokens: {response.tokens_used}")
            print(f"    Finish reason: {response.finish_reason}")
            if len(response.raw_response) > 0:
                print(f"    Raw response preview: {response.raw_response[:500]}...")

        return response.code or "", response.tokens_used

    def _refine_with_diagnostics(self, spec: KernelSpec, result: ExplorationResult) -> str:
        """Stage 7: Refine code based on diagnostics."""

        # Write initial code and test
        from experiments.generation.harness import write_kernel, check_correctness, _load_kernel_fn

        write_kernel(spec, result.initial_code)

        try:
            import torch
            kernel_fn = _load_kernel_fn(spec)

            # Check correctness
            correct, error_msg, _, _ = check_correctness(spec, kernel_fn, torch.float32, "cuda", False)

            if not correct:
                result.compilation_errors.append(error_msg)

                # Try to fix with AI
                from experiments.generation.exploration.prompts_exploration import build_refinement_prompt

                prompt = build_refinement_prompt(spec, result, error_msg)
                response = self.code_provider.generate_code(prompt, max_tokens=16384, temperature=0.0)
                result.total_tokens_used += response.tokens_used

                return response.code or result.initial_code

        except Exception as e:
            result.compilation_errors.append(str(e))
            # Try to recover with AI
            from experiments.generation.exploration.prompts_exploration import build_refinement_prompt

            prompt = build_refinement_prompt(spec, result, str(e))
            response = self.code_provider.generate_code(prompt, max_tokens=16384, temperature=0.0)
            result.total_tokens_used += response.tokens_used

            return response.code or result.initial_code

        return result.initial_code
