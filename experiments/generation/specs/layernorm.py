import torch

from experiments.generation.spec import KernelSpec, register_spec


def _reference_layernorm(x, weight, bias, eps):
    """PyTorch reference: LayerNorm forward."""
    mean = x.mean(dim=-1, keepdim=True)
    variance = x.var(dim=-1, keepdim=True, unbiased=False)
    x_normed = (x - mean) * torch.rsqrt(variance + eps)
    return x_normed * weight + bias


def _make_inputs(dtype=torch.float32, device="cuda",
                 batch=4, seq_len=512, hidden=768):
    x = torch.randn(batch, seq_len, hidden, dtype=dtype, device=device, requires_grad=True)
    weight = torch.ones(hidden, dtype=dtype, device=device, requires_grad=True)
    bias = torch.zeros(hidden, dtype=dtype, device=device, requires_grad=True)
    eps = 1e-5
    return (x, weight, bias, eps)


register_spec(KernelSpec(
    name="layernorm",
    description="Fused LayerNorm: normalize by mean/variance then affine transform",
    reference_fn=_reference_layernorm,
    input_factory=_make_inputs,
    check_backward=True,
    atol=1e-5,
    rtol=1e-5,
    target_file="ironcore/kernels/triton/layernorm.py",
    kernel_fn_name="triton_layernorm",
    input_sizes=[
        ("small", dict(batch=1, seq_len=128, hidden=768)),
        ("medium", dict(batch=4, seq_len=512, hidden=768)),
        ("large", dict(batch=8, seq_len=1024, hidden=2048)),
        ("xlarge", dict(batch=4, seq_len=2048, hidden=4096)),
    ],
))
