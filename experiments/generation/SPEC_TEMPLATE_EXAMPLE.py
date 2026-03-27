"""
Example: How to create a new kernel spec for the exploration framework.

Add your layer spec to experiments/generation/specs/<your_layer>.py
"""

import torch

from experiments.generation.spec import KernelSpec, register_spec, PerformanceTargets


# ============================================================
# Step 1: Define the reference implementation (PyTorch)
# ============================================================
def _reference_your_layer(x, weight, bias, eps):
    """
    PyTorch reference implementation of your layer.

    This should be the mathematically correct version that
    the Triton kernel will replicate.
    """
    # Example: Layer Normalization
    mean = x.mean(dim=-1, keepdim=True)
    var = x.var(dim=-1, keepdim=True, unbiased=False)
    x_normed = (x - mean) / torch.sqrt(var + eps)
    return x_normed * weight + bias


# ============================================================
# Step 2: Define input factory for testing
# ============================================================
def _make_inputs(
    dtype=torch.float32,
    device="cuda",
    batch=4,
    seq_len=512,
    hidden=768,
    **kwargs
):
    """
    Create test inputs for validation.

    The exploration framework will use these to:
    - Test correctness
    - Check gradients
    - Profile performance
    """
    x = torch.randn(batch, seq_len, hidden, dtype=dtype, device=device, requires_grad=True)
    weight = torch.ones(hidden, dtype=dtype, device=device, requires_grad=True)
    bias = torch.zeros(hidden, dtype=dtype, device=device, requires_grad=True)
    eps = 1e-5

    return (x, weight, bias, eps)


# ============================================================
# Step 3: Register the spec
# ============================================================
register_spec(KernelSpec(
    name="your_layer",              # Unique identifier
    description="Your layer description",
    reference_fn=_reference_your_layer,
    input_factory=_make_inputs,
    check_backward=True,              # Check gradient correctness
    atol=1e-5,                        # Absolute tolerance
    rtol=1e-5,                        # Relative tolerance
    target_file="ironcore/kernels/triton/your_layer.py",
    kernel_fn_name="triton_your_layer",
    input_sizes=[
        ("small", dict(batch=1, seq_len=128, hidden=768)),
        ("medium", dict(batch=4, seq_len=512, hidden=768)),
        ("large", dict(batch=8, seq_len=1024, hidden=2048)),
    ],
    performance_targets=PerformanceTargets(
        min_speedup=2.0,              # Target speedup over reference
    ),
    optimization_hints=[
        "Use two-pass approach: compute statistics first, then normalize",
        "Use scalar accumulators for reductions",
        "Ensure float32 precision for gradient accumulation",
    ],
))
