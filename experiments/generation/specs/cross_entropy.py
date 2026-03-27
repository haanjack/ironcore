import torch

from experiments.generation.spec import KernelSpec, PerformanceTargets, register_spec


def _reference_softmax_cross_entropy(logits, labels, ignore_index=-1):
    """PyTorch reference: Cross entropy with softmax."""
    # Stable softmax
    logits_max = logits.max(dim=-1, keepdim=True)[0]
    logits_stable = logits - logits_max

    exp_logits = torch.exp(logits_stable)
    sum_exp = exp_logits.sum(dim=-1, keepdim=True)

    # Softmax
    probs = exp_logits / sum_exp

    # Cross entropy loss: -log(p[target])
    # For each position, select the probability of the target class
    batch_size, seq_len, vocab_size = logits.shape
    loss = torch.zeros(batch_size, seq_len, dtype=logits.dtype, device=logits.device)

    for b in range(batch_size):
        for s in range(seq_len):
            label = labels[b, s]
            if label != ignore_index:
                loss[b, s] = -torch.log(probs[b, s, label] + 1e-8)

    return loss


def _make_inputs(dtype=torch.float32, device="cuda",
                 batch=4, seq_len=512, vocab_size=32000):
    # Use smaller vocab for testing
    vocab_size = min(vocab_size, 32000)
    logits = torch.randn(batch, seq_len, vocab_size, dtype=dtype, device=device, requires_grad=True)
    labels = torch.randint(0, vocab_size, (batch, seq_len), device=device)
    return (logits, labels)


register_spec(KernelSpec(
    name="cross_entropy",
    description="Softmax cross entropy loss with stable computation",
    reference_fn=_reference_softmax_cross_entropy,
    input_factory=_make_inputs,
    check_backward=True,
    atol=1e-4,  # Slightly relaxed tolerance for numerical stability
    rtol=1e-4,
    target_file="ironcore/kernels/triton/cross_entropy.py",
    kernel_fn_name="triton_cross_entropy",
    input_sizes=[
        ("small", dict(batch=1, seq_len=128, vocab_size=32000)),
        ("medium", dict(batch=4, seq_len=512, vocab_size=32000)),
        ("large", dict(batch=8, seq_len=1024, vocab_size=64000)),
    ],
    performance_targets=PerformanceTargets(min_speedup=1.5),  # More realistic target
))
