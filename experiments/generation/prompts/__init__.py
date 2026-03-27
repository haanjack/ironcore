"""Prompt system for AI code generation.

This module provides Jinja2-based templates for generating prompts
for AI-powered kernel generation and optimization.

Usage:
    from experiments.generation.prompts import build_prompt, build_optimization_prompt

    # Initial generation
    prompt = build_prompt(spec, backend="triton")

    # Optimization based on profiling
    opt_prompt = build_optimization_prompt(spec, validation_result, analysis, current_code)
"""

import inspect
from pathlib import Path
from typing import Optional

from jinja2 import Environment, FileSystemLoader, Template

from experiments.generation.spec import KernelSpec

PROMPT_DIR = Path(__file__).parent
_env = Environment(loader=FileSystemLoader(PROMPT_DIR))


def load_prompt(template_name: str) -> Template:
    """Load a prompt template by name.

    Args:
        template_name: Name of the template file (e.g., "base.j2")

    Returns:
        Jinja2 Template object
    """
    return _env.get_template(template_name)


def build_prompt(
    spec: KernelSpec,
    backend: str = "triton",
    example_kernel_path: Optional[str] = None,
) -> str:
    """Build a generation prompt for a kernel spec.

    Args:
        spec: Kernel specification
        backend: Backend type (triton, tilelang)
        example_kernel_path: Optional path to example kernel for reference

    Returns:
        Formatted prompt string
    """
    template = load_prompt("base.j2")

    # Get reference function source
    reference_source = inspect.getsource(spec.reference_fn)

    # Build input signature from reference function
    input_signature = _build_input_signature(spec)

    # Load example kernel if provided
    example_kernel = ""
    if example_kernel_path:
        example_path = Path(example_kernel_path)
        if example_path.exists():
            example_kernel = example_path.read_text()

    # Render template
    prompt = template.render(
        spec=spec,
        backend=backend,
        reference_source=reference_source,
        input_signature=input_signature,
        example_kernel=example_kernel,
    )

    return prompt


def _build_input_signature(spec: KernelSpec) -> str:
    """Build a readable input signature from the spec.

    Args:
        spec: Kernel specification

    Returns:
        String representation of input signature
    """
    # Try to get the signature from the reference function
    try:
        sig = inspect.signature(spec.reference_fn)
        params = []
        for name, param in sig.parameters.items():
            if param.default == inspect.Parameter.empty:
                params.append(name)
            else:
                params.append(f"{name}={param.default}")
        return ", ".join(params)
    except Exception:
        # Fallback to generic description
        return "Inputs matching reference function signature"


def build_refine_prompt(
    spec: KernelSpec,
    validation_result,
    current_code: str,
    error_msg: str,
) -> str:
    """Build a refinement prompt for fixing validation errors.

    Args:
        spec: Kernel specification
        validation_result: ValidationResult from harness
        current_code: Current generated code
        error_msg: Error message from validation

    Returns:
        Formatted refinement prompt
    """
    # Add kernel-specific numerical hints
    numerical_hints = ""
    if "cross_entropy" in spec.name or "softmax" in spec.name:
        numerical_hints = """
## NUMERICAL PRECISION HINTS (for softmax/cross_entropy)
The most common cause of numerical errors is precision loss in exp/sum operations:

1. **Always convert to float32 when loading**:
   ```python
   x = tl.load(ptr + offsets, mask=mask).to(tl.float32)
   ```

2. **Use max-subtraction trick for exp** (prevents overflow):
   ```python
   max_val = tl.max(x, axis=0)  # Find max first
   exp_vals = tl.exp(x - max_val)  # Then exp(x - max)
   sum_exp = tl.sum(exp_vals, axis=0)
   ```

3. **For log-sum-exp, use stable formula**:
   ```python
   log_sum_exp = max_val + tl.log(sum_exp)  # NOT tl.log(tl.sum(tl.exp(x)))
   ```

4. **Initialize accumulators as float**:
   ```python
   sum_exp = 0.0  # NOT sum_exp = 0
   ```

5. **For large reductions, mask out invalid elements**:
   ```python
   exp_vals = tl.where(mask, exp_vals, 0.0)  # Don't include masked in sum
   ```
"""

    prompt = f"""The previous kernel generation failed validation. Fix the following issues:

## Kernel Specification
Name: {spec.name}
Description: {spec.description}

## Validation Error
{error_msg}

## CRITICAL REQUIREMENTS
- Export function name: {spec.kernel_fn_name}
- You MUST create a function named `{spec.kernel_fn_name}` at module level
- At the end of your code, you MUST include: `{spec.kernel_fn_name} = your_function_name`
- Numerical tolerance: atol={spec.atol}, rtol={spec.rtol}
- Gradient support: {"Yes" if spec.check_backward else "No"}
{numerical_hints}
## Previous Code (fix this)
```python
{current_code[:3000]}
```

## Instructions
Fix the errors in the code above. Make sure to:
1. Create a function named `{spec.kernel_fn_name}` at module level
2. At the end, assign your main function like: `{spec.kernel_fn_name} = YourClassName.apply` or `{spec.kernel_fn_name} = your_function_name`
3. Do NOT use Python's ** operator for power in Triton code - use multiplication (e.g., x * x * x) or tl.pow(x, 3)
4. Pay attention to numerical precision - use .to(tl.float32) for all computations

Return ONLY the corrected Python code without markdown code blocks or explanations.
"""
    return prompt


def build_optimization_prompt(
    spec: KernelSpec,
    validation_result,
    analysis,
    current_code: str,
) -> str:
    """Build an optimization prompt based on profiling analysis.

    Args:
        spec: Kernel specification
        validation_result: ValidationResult from harness
        analysis: OptimizationAnalysis from profiling analyzer
        current_code: Current kernel implementation

    Returns:
        Formatted optimization prompt
    """
    template = load_prompt("optimization.j2")

    # Get reference function source
    reference_source = inspect.getsource(spec.reference_fn)

    # Render template
    prompt = template.render(
        spec=spec,
        validation_result=validation_result,
        analysis=analysis,
        current_code=current_code,
        reference_source=reference_source,
    )

    return prompt


def get_reference_source(spec: KernelSpec) -> str:
    """Get the source code of the reference function.

    Args:
        spec: Kernel specification

    Returns:
        Source code as string
    """
    return inspect.getsource(spec.reference_fn)
