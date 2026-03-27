import torch

from experiments.generation.spec import KernelSpec, register_spec


def _reference_safe_softmax(x):
    """PyTorch reference: numerically stable softmax along last dimension."""
    max_vals = x.max(dim=-1, keepdim=True)[0]
    x_shifted = x - max_vals
    exp_x = torch.exp(x_shifted)
    return exp_x / exp_x.sum(dim=-1, keepdim=True)


def _make_inputs(dtype=torch.float32, device="cuda",
                 batch=4, num_heads=12, seq_len=512):
    # Attention score shape: [batch, heads, seq_q, seq_k]
    x = torch.randn(batch, num_heads, seq_len, seq_len, dtype=dtype, device=device, requires_grad=True)
    return (x,)


register_spec(KernelSpec(
    name="softmax",
    description="Fused safe softmax: max-subtract + exp + normalize in one pass",
    reference_fn=_reference_safe_softmax,
    input_factory=_make_inputs,
    check_backward=True,
    atol=1e-5,
    rtol=1e-5,
    target_file="ironcore/kernels/triton/softmax.py",
    kernel_fn_name="triton_softmax",
    input_sizes=[
        ("small", dict(batch=2, num_heads=12, seq_len=128)),
        ("medium", dict(batch=4, num_heads=12, seq_len=512)),
        ("large", dict(batch=2, num_heads=32, seq_len=1024)),
    ],
))
