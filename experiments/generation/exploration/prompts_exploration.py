"""
Prompts for the exploration phase of AI-driven Triton kernel generation.

Each stage has a specific prompt that guides the AI through a focused analysis.
"""

import inspect
from experiments.generation.spec import KernelSpec


def build_algorithm_analysis_prompt(spec: KernelSpec) -> str:
    """Stage 1: Analyze the algorithm and understand its structure."""

    reference_source = inspect.getsource(spec.reference_fn)

    return f"""You are a GPU kernel architecture expert. Analyze the following algorithm for Triton GPU implementation.

## Kernel Specification
Name: {spec.name}
Description: {spec.description}

## Reference Implementation (PyTorch)
```python
{reference_source}
```

## Task
Provide a structured analysis covering:

1. **Algorithm Structure**
   - What is the mathematical operation?
   - What are the inputs and outputs?
   - Are there multiple computational passes?

2. **Computational Characteristics**
   - Element-wise operations
   - Reduction operations (sum, max, min, etc.)
   - Dependencies between operations

3. **Memory Access Patterns**
   - How is data read?
   - How is data written?
   - Any special memory requirements?

4. **Parallelization Strategy**
   - What can be parallelized?
   - What needs synchronization?
   - Any atomic operations needed?

5. **Gradient Computation (if applicable)**
   - What intermediates need to be saved?
   - What is the gradient complexity?

Provide your analysis as a structured text response (not code).
"""


def build_graph_analysis_prompt(spec: KernelSpec) -> str:
    """Stage 2: Analyze computation graph and data flow."""

    return f"""You are analyzing the computation graph for GPU kernel implementation.

## Kernel: {spec.name}

## Task
Create a detailed computation graph analysis:

1. **Input Tensors**
   - Shapes and dtypes
   - Memory layout (contiguous/strided)
   - Access patterns

2. **Computational Graph**
   - List each operation node
   - Identify reduction nodes
   - Identify element-wise nodes
   - Data dependencies between nodes

3. **Memory Access Analysis**
   - Reads per element
   - Writes per element
   - Optimal access pattern

4. **Parallelization Opportunities**
   - Can rows be processed independently? (yes/no)
   - Can columns be processed independently? (yes/no)
   - Are there reductions that need atomic ops? (yes/no)

5. **Classification**
   - Is this primarily: element-wise / reduction / scan / complex?

Return analysis in structured format.
"""


def build_tiling_analysis_prompt(spec: KernelSpec, graph_analysis) -> str:
    """Stage 3: Determine optimal tiling/block size strategy."""

    return f"""You are determining the optimal tiling strategy for a Triton kernel.

## Kernel: {spec.name}

## Graph Analysis
- Operations: {graph_analysis.operations}
- Reductions: {graph_analysis.reductions}
- Can parallelize rows: {graph_analysis.can_parallelize_rows}
- Can parallelize cols: {graph_analysis.can_parallelize_cols}
- Requires reduction: {graph_analysis.requires_reduction}

## Task
Provide tiling strategy analysis:

1. **Block Size Recommendation**
   - Recommended BLOCK_SIZE
   - Minimum and maximum viable sizes
   - Rationale for this choice

2. **Memory Hierarchy Considerations**
   - Will the working set fit in L1 cache?
   - Will it fit in registers?
   - Expected register pressure: low/medium/high

3. **Alternative Strategies**
   - What if BLOCK_SIZE is too small?
   - What if BLOCK_SIZE is too large?
   - Any need for multi-tiling (e.g., 2D tiling)?

4. **Special Considerations**
   - Loop unrolling considerations
   - Vectorization width
   - num_warps recommendation

Return analysis with specific recommendations.
"""


def build_conversion_plan_prompt(spec: KernelSpec, result) -> str:
    """Stage 4: Create structured conversion plan from algorithm to Triton."""

    passes_str = "\\n".join([
        f"Pass {i+1}: {p.get('name', 'unnamed')} ({p.get('type', 'unknown')})"
        for i, p in enumerate(result.conversion_plan.passes)
    ])

    return f"""You are creating a detailed implementation plan for converting an algorithm to Triton.

## Kernel: {spec.name}

## Algorithm Summary
{result.algorithm_summary}

## Tiling Strategy
- Block size: {result.tiling_strategy.recommended_block_size}
- Rationale: {result.tiling_strategy.tiling_rationale}

## Task
Create a step-by-step conversion plan:

1. **Kernel Structure**
   - Name: main_forward_kernel, main_backward_kernel
   - Input/output pointer parameters
   - Stride parameters (for each tensor dimension)
   - Compile-time parameters (BLOCK_SIZE, etc.)

2. **Computational Passes**
{passes_str}

3. **Memory Management**
   - Are intermediate buffers needed?
   - What to save for backward pass?
   - Atomic operations needed?

4. **Gradient Strategy** (if applicable)
   - Saved tensors from forward
   - Gradient computation approach
   - Complexity: simple / medium / complex

Provide plan as numbered steps.
"""


def build_structure_design_prompt(spec: KernelSpec, result) -> str:
    """Stage 5: Design the detailed kernel code structure."""

    passes_str = "\\n".join([
        f"Pass {i+1}: {p.get('name', 'unnamed')} ({p.get('type', 'unknown')})"
        for i, p in enumerate(result.conversion_plan.passes)
    ])

    return f"""You are designing the code structure for a Triton kernel implementation.

## Kernel: {spec.name}

## Conversion Plan
{passes_str}

## Task
Design the detailed code structure:

1. **Kernel Function Signatures**
   ```python
   @triton.jit
   def forward_kernel(..., BLOCK_SIZE: tl.constexpr):
       # Implementation structure
   ```

   Define:
   - All pointer parameters
   - All stride/shape parameters
   - All compile-time constants

2. **Body Structure**
   For each pass:
   - Loop structure (for tiling)
   - Memory loads (with proper masking)
   - Computation steps
   - Memory stores

3. **Python Wrapper Structure**
   ```python
   class MyFunction(torch.autograd.Function):
       @staticmethod
       def forward(ctx, ...):
           # Reshape for arbitrary input
           # Launch kernel
           # Save for backward
           # Return reshaped output

       @staticmethod
       def backward(ctx, grad_output):
           # Load saved tensors
           # Reshape grad_output
           # Launch backward kernel
           # Return gradients
   ```

4. **Critical Implementation Notes**
   - Shape handling (reshape to 2D)
   - Numerical precision (use float32 accumulators)
   - Edge cases (masking, padding)

Provide code structure as Python code with comments.
"""


def build_initial_code_prompt(spec: KernelSpec, result) -> str:
    """Stage 6: Generate the initial complete Triton kernel code."""

    reference_source = inspect.getsource(spec.reference_fn)

    passes_str = "\\n".join([
        f"- {p.get('name', 'unnamed')}"
        for p in result.conversion_plan.passes
    ])

    return f"""You are an expert Triton kernel developer. Generate a complete, working Triton kernel.

## Kernel Specification
Name: {spec.name}
Description: {spec.description}

## Reference Implementation (PyTorch)
```python
{reference_source}
```

## Exploration Analysis

### Algorithm Summary
{result.algorithm_summary}

### Tiling Strategy
- Recommended BLOCK_SIZE: {result.tiling_strategy.recommended_block_size}
- Register pressure: {result.tiling_strategy.register_pressure}

### Conversion Plan
{passes_str}

## Implementation Requirements

### 1. CRITICAL: Code Must Start With
```python
import torch
import triton
import triton.language as tl
```

### 2. Numerical Precision Rules
- Use `x * x` for squaring (NOT `x ** 2`, NOT `tl.pow()`)
- Use scalar accumulators: `acc = 0.0` then `acc += tl.sum(x, axis=0)`
- For weight gradients: use `torch.zeros(N, dtype=torch.float32)` buffer

### 3. Shape Handling
```python
# In forward:
original_shape = x.shape
x_reshaped = x.reshape(-1, x.shape[-1]).contiguous()
# ... process ...
return output.reshape(original_shape)

# In backward:
grad_output_reshaped = grad_output.reshape(-1, grad_output.shape[-1]).contiguous()
# ... process ...
return grad_x.reshape(original_shape), ...
```

### 4. Structure Template
```python
import torch
import triton
import triton.language as tl

@triton.jit
def forward_kernel(
    # Pointer parameters
    # Stride parameters
    # Shape parameters
    # Compile-time constants
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)

    # Pass 1: ...
    for off in range(0, n_cols, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < n_cols
        # Load, compute, store

    # Pass 2: ...
    # ...

class MyFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, ...):
        original_shape = x.shape
        x_reshaped = x.reshape(-1, x.shape[-1]).contiguous()
        M, N = x_reshaped.shape

        BLOCK_SIZE = triton.next_power_of_2(N)
        if BLOCK_SIZE > 8192:
            BLOCK_SIZE = 8192

        grid = (M,)

        forward_kernel[grid](
            # Parameters
            BLOCK_SIZE=BLOCK_SIZE
        )

        ctx.save_for_backward(...)
        ctx.original_shape = original_shape
        return output.reshape(original_shape)

    @staticmethod
    def backward(ctx, grad_output):
        x_reshaped, ... = ctx.saved_tensors
        original_shape = ctx.original_shape

        grad_output_reshaped = grad_output.reshape(-1, grad_output.shape[-1]).contiguous()

        # Launch backward kernel

        return grad_x.reshape(original_shape), ...

def {spec.kernel_fn_name}(...):
    return MyFunction.apply(...)
```

### 5. Common Patterns

**Scalar Accumulator (for reductions):**
```python
# CORRECT
acc = 0.0
for off in range(0, n_cols, BLOCK_SIZE):
    x = tl.load(...)
    acc += tl.sum(x, axis=0)  # Reduce to scalar FIRST
result = acc / n_cols
```

**Float32 Gradient Buffer:**
```python
# In Python wrapper
dw = torch.zeros(N, dtype=torch.float32, device=device)
# ... kernel with tl.atomic_add ...
# After backward
if w.dtype != torch.float32:
    dw = dw.to(w.dtype)
```

## Generate Complete Kernel Now

Return ONLY valid Python code. No markdown, no explanations, no placeholders.

Generate the complete kernel:
"""


def build_refinement_prompt(spec: KernelSpec, result, error_msg: str) -> str:
    """Stage 7: Refine code based on diagnostic feedback."""

    return f"""You are refining a Triton kernel based on error diagnostics.

## Kernel: {spec.name}

## Current Code
```python
{result.initial_code}
```

## Diagnostic Error
{error_msg}

## Task
Analyze the error and provide corrected code.

### Error Analysis Guidelines

1. **Compilation Errors**
   - Check for undefined functions (tl.pow doesn't exist, use x*x)
   - Check for incorrect parameter types
   - Check for missing imports

2. **Numerical Errors**
   - High max_diff (> 1e-4): Check accumulator type (use scalar, not vector)
   - Systematic bias: Check reduction order
   - Sparse large errors: Check boundary conditions/masking

3. **Gradient Errors**
   - dw/db errors: Check float32 buffer usage
   - dx errors: Check gradient chain computation

### Common Fixes

**Issue: Missing import torch**
```python
# Add at top:
import torch
import triton
import triton.language as tl
```

**Issue: Vector accumulator**
```python
# WRONG
_sum = tl.zeros([BLOCK_SIZE])
_sum += x
result = tl.sum(_sum) / N

# CORRECT
result = 0.0
for off in range(0, N, BLOCK_SIZE):
    x = tl.load(...)
    result += tl.sum(x, axis=0)
result /= N
```

**Issue: Gradient precision**
```python
# In backward method:
dw = torch.zeros(N, dtype=torch.float32, device=device)  # fp32 buffer
# ... launch kernel ...
if w.dtype != torch.float32:
    dw = dw.to(w.dtype)
```

Provide the corrected, complete kernel code.
"""
