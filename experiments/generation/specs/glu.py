import torch

from experiments.generation.spec import KernelSpec, register_spec


def _reference_swiglu(x):
    """PyTorch reference: SwiGLU forward."""
    x_feature, x_gate = x.chunk(2, dim=-1)
    return torch.nn.functional.silu(x_feature) * x_gate


def _make_inputs(dtype=torch.float32, device="cuda",
                 batch=4, seq_len=512, hidden=4096):
    x = torch.randn(batch, seq_len, hidden * 2, dtype=dtype, device=device, requires_grad=True)
    return (x,)


register_spec(KernelSpec(
    name="swiglu",
    description="SwiGLU activation: silu(x_feature) * x_gate",
    reference_fn=_reference_swiglu,
    input_factory=_make_inputs,
    check_backward=True,
    atol=1e-5,
    rtol=1e-5,
    target_file="ironcore/kernels/triton/glu.py",
    kernel_fn_name="triton_swiglu",
    input_sizes=[
        ("small", dict(batch=1, seq_len=128, hidden=768)),
        ("medium", dict(batch=4, seq_len=512, hidden=2048)),
        ("large", dict(batch=8, seq_len=1024, hidden=4096)),
    ],
))


def _reference_glu(x):
    """PyTorch reference: GLU forward."""
    x_feature, x_gate = x.chunk(2, dim=-1)
    return x_feature * torch.sigmoid(x_gate)


def _make_inputs_glu(dtype=torch.float32, device="cuda",
                     batch=4, seq_len=512, hidden=4096):
    x = torch.randn(batch, seq_len, hidden * 2, dtype=dtype, device=device, requires_grad=True)
    return (x,)


register_spec(KernelSpec(
    name="glu",
    description="GLU activation: x_feature * sigmoid(x_gate)",
    reference_fn=_reference_glu,
    input_factory=_make_inputs_glu,
    check_backward=True,
    atol=1e-5,
    rtol=1e-5,
    target_file="ironcore/kernels/triton/glu.py",
    kernel_fn_name="triton_glu",
    input_sizes=[
        ("small", dict(batch=1, seq_len=128, hidden=768)),
        ("medium", dict(batch=4, seq_len=512, hidden=2048)),
        ("large", dict(batch=8, seq_len=1024, hidden=4096)),
    ],
))
