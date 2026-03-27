import torch

from experiments.generation.spec import KernelSpec, register_spec


def _reference_rmsnorm(x, weight, eps):
    """PyTorch reference: RMSNorm forward."""
    variance = x.pow(2).mean(dim=-1, keepdim=True)
    x_normed = x * torch.rsqrt(variance + eps)
    return x_normed * weight


def _make_inputs(dtype=torch.float32, device="cuda",
                 batch=4, seq_len=512, hidden=768):
    x = torch.randn(batch, seq_len, hidden, dtype=dtype, device=device, requires_grad=True)
    weight = torch.ones(hidden, dtype=dtype, device=device, requires_grad=True)
    eps = 1e-5
    return (x, weight, eps)


register_spec(KernelSpec(
    name="rmsnorm",
    description="Fused RMSNorm: normalize by RMS then scale by weight",
    reference_fn=_reference_rmsnorm,
    input_factory=_make_inputs,
    check_backward=True,
    atol=1e-5,
    rtol=1e-5,
    target_file="ironcore/kernels/triton/rmsnorm.py",
    kernel_fn_name="triton_rmsnorm",
    input_sizes=[
        ("small", dict(batch=1, seq_len=128, hidden=768)),
        ("medium", dict(batch=4, seq_len=512, hidden=768)),
        ("large", dict(batch=8, seq_len=1024, hidden=2048)),
        ("xlarge", dict(batch=4, seq_len=2048, hidden=4096)),
    ],
))
