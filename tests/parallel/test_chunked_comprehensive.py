"""
Comprehensive validation for chunked tensor parallelism with TP=1 and TP=2.

Compares baseline (no chunking) against chunked execution across:
- TP sizes: 1 (single GPU) and 2 (2 GPUs with async all-reduce)
- Multiple chunk configurations (2, 4, 8 chunks)
- Short, medium, and long sequences
- Standard attention and flash attention
- Forward activations, loss values, and gradient norms
- Memory footprints (allocation and reservation)

Usage:
    # Test TP=1 (single GPU)
    python -m pytest tests/test_chunked_validation.py -v

    # Test TP=2 (2 GPUs) - requires manual invocation
    torchrun --nproc_per_node=2 tests/test_chunked_validation.py TestChunkedValidationTP2
"""

import argparse
import os
import sys
import unittest

import torch
import torch.distributed as dist

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ironcore.config import (
    DataConfig,
    InitConfig,
    MainConfig,
    ModelConfig,
    OperationConfig,
    OptimConfig,
    ParallelConfig,
    TrainerConfig,
    UtilsConfig,
)
from ironcore.models.transformer import TransformerModel
from ironcore.parallel import parallel_states

D_MODEL = 128
NUM_HEADS = 4
NUM_GROUPS = 4
HEAD_DIM = D_MODEL // NUM_HEADS
D_FFN = 256
NUM_LAYERS = 2
BATCH_SIZE = 2
SEED = 42


def create_config(
    seq_len=128,
    use_flash_attn=False,
    sequence_chunk_size=None,
    precision="float32",
    no_bias=False,
    tp_size=1,
):
    return MainConfig(
        model=ModelConfig(
            d_model=D_MODEL,
            num_attention_heads=NUM_HEADS,
            num_attention_groups=NUM_GROUPS,
            head_dim=HEAD_DIM,
            d_ffn=D_FFN,
            num_layers=NUM_LAYERS,
            max_seq_len=seq_len,
            max_position_embeddings=seq_len,
            dropout_attn=0.0,
            dropout_mlp=0.0,
            dropout_embd=0.0,
            no_bias=no_bias,
            precision=precision,
        ),
        trainer=TrainerConfig(
            tensor_model_parallel_size=tp_size,
            use_flash_attn=use_flash_attn,
            sequence_chunk_size=sequence_chunk_size,
        ),
        init=InitConfig(seed=SEED, init_std=0.02),
        optim=OptimConfig(),
        data=DataConfig(),
        parallel=ParallelConfig(timeout_minute=30),
        operation=OperationConfig(),
        utils=UtilsConfig(),
    )


def run_forward_backward(model, hidden_states, attention_mask, config, chunk_size, device):
    """Run forward + backward, return activations, loss, gradients, and memory stats."""
    config.trainer.sequence_chunk_size = chunk_size
    model.zero_grad()

    # Reset memory stats before measurement
    torch.cuda.reset_peak_memory_stats(device)

    x = hidden_states.clone().detach().requires_grad_(True)
    output = model(x, attention_mask, rotary_pos_emb=None)

    loss = output.pow(2).mean()
    loss.backward()

    # Capture memory stats after backward
    peak_alloc = torch.cuda.max_memory_allocated(device) / (1024**2)  # MiB
    peak_resv = torch.cuda.max_memory_reserved(device) / (1024**2)

    param_grads = {}
    for name, p in model.named_parameters():
        if p.grad is not None:
            param_grads[name] = p.grad.detach().clone()

    return {
        "output": output.detach().clone(),
        "loss": loss.item(),
        "input_grad": x.grad.detach().clone(),
        "param_grads": param_grads,
        "peak_alloc_mib": peak_alloc,
        "peak_resv_mib": peak_resv,
    }


def compare(baseline, result, atol, rtol, label):
    """Compare two runs and return a dict of metrics + pass/fail."""
    b_out, r_out = baseline["output"], result["output"]
    out_abs = (b_out - r_out).abs()
    max_out_diff = out_abs.max().item()
    out_norm_base = b_out.norm().item()
    out_norm_chunk = r_out.norm().item()

    loss_diff = abs(baseline["loss"] - result["loss"])

    # input gradient
    ig_diff = (baseline["input_grad"] - result["input_grad"]).abs().max().item()

    # parameter gradients
    max_grad_diff = 0.0
    worst_param = ""
    grad_norm_rows = []
    for name in baseline["param_grads"]:
        gb = baseline["param_grads"][name]
        gc = result["param_grads"][name]
        diff = (gb - gc).abs().max().item()
        bn = gb.norm().item()
        cn = gc.norm().item()
        nd = abs(bn - cn) / (bn + 1e-12)
        if diff > max_grad_diff:
            max_grad_diff = diff
            worst_param = name
        grad_norm_rows.append((name, bn, cn, nd, diff))

    # Memory comparison
    mem_delta = result["peak_alloc_mib"] - baseline["peak_alloc_mib"]

    passed = max_out_diff < atol and loss_diff < atol and max_grad_diff < atol

    return {
        "label": label,
        "max_output_diff": max_out_diff,
        "output_norm_base": out_norm_base,
        "output_norm_chunk": out_norm_chunk,
        "loss_base": baseline["loss"],
        "loss_chunk": result["loss"],
        "loss_diff": loss_diff,
        "input_grad_diff": ig_diff,
        "max_grad_diff": max_grad_diff,
        "worst_param": worst_param,
        "grad_norm_rows": grad_norm_rows,
        "peak_alloc_base": baseline["peak_alloc_mib"],
        "peak_alloc_chunk": result["peak_alloc_mib"],
        "mem_delta": mem_delta,
        "passed": passed,
        "atol": atol,
    }


def print_report(metrics_list, section_title):
    """Pretty-print a comparison table."""
    print(f"\n{'=' * 100}")
    print(f"  {section_title}")
    print(f"{'=' * 100}")

    header = f"{'Config':<28} {'Out Diff':>10} {'Loss Diff':>10} {'Grad Diff':>10} {'Mem Δ':>9} {'Pass':>6}"
    print(header)
    print("-" * 100)

    for m in metrics_list:
        status = "✓ OK" if m["passed"] else "✗ FAIL"
        print(
            f"{m['label']:<28} "
            f"{m['max_output_diff']:>10.2e} "
            f"{m['loss_diff']:>10.2e} "
            f"{m['max_grad_diff']:>10.2e} "
            f"{m['mem_delta']:>+8.0f}MB "
            f"{status:>6}"
        )

    # detailed gradient norms for the last entry (longest sequence / most chunks)
    m = metrics_list[-1]
    print(f"\n  Gradient norm detail for [{m['label']}]:")
    print(
        f"  {'Parameter':<45} {'Base Norm':>11} {'Chunk Norm':>11} {'Rel Diff':>10} {'Abs Diff':>10}"
    )
    print(f"  {'-' * 90}")
    for name, bn, cn, nd, ad in m["grad_norm_rows"]:
        print(f"  {name:<45} {bn:>11.6f} {cn:>11.6f} {nd:>10.2e} {ad:>10.2e}")

    print(
        f"\n  Activation norms: baseline={m['output_norm_base']:.6f}  chunked={m['output_norm_chunk']:.6f}"
    )
    print(f"  Loss values:      baseline={m['loss_base']:.8f}  chunked={m['loss_chunk']:.8f}")
    print(f"  Memory usage:     baseline={m['peak_alloc_base']:.0f}MB  chunked={m['peak_alloc_chunk']:.0f}MB  Δ={m['mem_delta']:+.0f}MB")


class TestChunkedValidation(unittest.TestCase):
    """Test chunked tensor parallelism with TP=1 (single GPU)."""

    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA not available")
        os.environ.setdefault("MASTER_ADDR", "localhost")
        os.environ.setdefault("MASTER_PORT", "12355")
        if not dist.is_initialized():
            dist.init_process_group(backend="nccl", rank=0, world_size=1)
        try:
            parallel_states.initialize_model_parallel(
                tensor_model_parallel_size=1,
                timeout_in_minutes=1.0,
            )
        except Exception:
            pass
        cls.device = torch.device("cuda:0")

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    def _run_suite(self, seq_len, chunk_sizes, dtype, use_flash_attn, no_bias, section, tp_size=1):
        precision = "float32" if dtype == torch.float32 else "bfloat16"
        config = create_config(
            seq_len=seq_len,
            use_flash_attn=use_flash_attn,
            precision=precision,
            no_bias=no_bias,
            tp_size=tp_size,
        )

        torch.manual_seed(SEED)
        torch.cuda.manual_seed(SEED)
        model = TransformerModel(config).to(device=self.device, dtype=dtype)
        model.init_weights()
        model.train()

        hidden = torch.randn(BATCH_SIZE, seq_len, D_MODEL, device=self.device, dtype=dtype)
        mask = torch.tril(torch.ones(1, 1, seq_len, seq_len, device=self.device, dtype=dtype))

        baseline = run_forward_backward(model, hidden, mask, config, chunk_size=None, device=self.device)

        atol = 1e-5 if dtype == torch.float32 else 5e-2
        rtol = 1e-4 if dtype == torch.float32 else 1e-1

        metrics = []
        for cs in chunk_sizes:
            n_chunks = (seq_len + cs - 1) // cs
            label = f"seq={seq_len} chunk={cs} (x{n_chunks})"
            result = run_forward_backward(model, hidden, mask, config, chunk_size=cs, device=self.device)
            m = compare(baseline, result, atol, rtol, label)
            metrics.append(m)

        print_report(metrics, section)

        for m in metrics:
            self.assertTrue(
                m["passed"],
                f"FAILED {m['label']}: out={m['max_output_diff']:.2e} "
                f"loss={m['loss_diff']:.2e} grad={m['max_grad_diff']:.2e} (atol={atol})",
            )

    # ------------------------------------------------------------------
    # Standard attention, float32
    # ------------------------------------------------------------------
    def test_short_seq_standard(self):
        self._run_suite(
            64,
            [32, 16, 8],
            torch.float32,
            False,
            False,
            "Standard Attn | fp32 | Short seq=64 | bias=True",
        )

    def test_medium_seq_standard(self):
        self._run_suite(
            256,
            [128, 64, 32],
            torch.float32,
            False,
            False,
            "Standard Attn | fp32 | Medium seq=256 | bias=True",
        )

    def test_long_seq_standard(self):
        self._run_suite(
            1024,
            [512, 256, 128],
            torch.float32,
            False,
            False,
            "Standard Attn | fp32 | Long seq=1024 | bias=True",
        )

    def test_long_seq_standard_no_bias(self):
        self._run_suite(
            1024,
            [512, 256, 128],
            torch.float32,
            False,
            True,
            "Standard Attn | fp32 | Long seq=1024 | bias=False",
        )

    def test_uneven_chunks_standard(self):
        self._run_suite(
            100,
            [33, 17, 13],
            torch.float32,
            False,
            False,
            "Standard Attn | fp32 | Uneven seq=100 | bias=True",
        )

    # ------------------------------------------------------------------
    # Flash attention, bfloat16
    # ------------------------------------------------------------------
    def test_short_seq_flash(self):
        try:
            from flash_attn import flash_attn_varlen_func  # noqa: F401
        except ImportError:
            self.skipTest("flash_attn not installed")
        self._run_suite(
            64,
            [32, 16, 8],
            torch.bfloat16,
            True,
            False,
            "Flash Attn | bf16 | Short seq=64 | bias=True",
        )

    def test_long_seq_flash(self):
        try:
            from flash_attn import flash_attn_varlen_func  # noqa: F401
        except ImportError:
            self.skipTest("flash_attn not installed")
        self._run_suite(
            1024,
            [512, 256, 128],
            torch.bfloat16,
            True,
            False,
            "Flash Attn | bf16 | Long seq=1024 | bias=True",
        )

    def test_vlong_seq_flash(self):
        try:
            from flash_attn import flash_attn_varlen_func  # noqa: F401
        except ImportError:
            self.skipTest("flash_attn not installed")
        self._run_suite(
            2048,
            [1024, 512, 256],
            torch.bfloat16,
            True,
            False,
            "Flash Attn | bf16 | VLong seq=2048 | bias=True",
        )


class TestChunkedValidationTP2(unittest.TestCase):
    """Test chunked tensor parallelism with TP=2 (2 GPUs with async all-reduce)."""

    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA not available")

        if torch.cuda.device_count() < 2:
            raise unittest.SkipTest("TP=2 requires at least 2 GPUs")

        if not dist.is_initialized():
            dist.init_process_group(backend="nccl")

        rank = dist.get_rank()
        world_size = dist.get_world_size()

        if world_size != 2:
            raise unittest.SkipTest(f"TP=2 tests require exactly 2 processes, got {world_size}")

        # Set CUDA device for this rank
        torch.cuda.set_device(rank)
        cls.device = torch.device(f"cuda:{rank}")

        parallel_states.initialize_model_parallel(
            tensor_model_parallel_size=2,
            timeout_in_minutes=30,
        )

    def _run_suite(self, seq_len, chunk_sizes, dtype, use_flash_attn, no_bias, section):
        rank = dist.get_rank()

        precision = "float32" if dtype == torch.float32 else "bfloat16"
        config = create_config(
            seq_len=seq_len,
            use_flash_attn=use_flash_attn,
            precision=precision,
            no_bias=no_bias,
            tp_size=2,
        )

        torch.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)
        model = TransformerModel(config).to(device=self.device, dtype=dtype)
        model.init_weights()
        model.train()

        hidden = torch.randn(BATCH_SIZE, seq_len, D_MODEL, device=self.device, dtype=dtype)
        mask = torch.tril(torch.ones(1, 1, seq_len, seq_len, device=self.device, dtype=dtype))

        baseline = run_forward_backward(model, hidden, mask, config, chunk_size=None, device=self.device)

        atol = 1e-5 if dtype == torch.float32 else 5e-2
        rtol = 1e-4 if dtype == torch.float32 else 1e-1

        metrics = []
        for cs in chunk_sizes:
            n_chunks = (seq_len + cs - 1) // cs
            label = f"seq={seq_len} chunk={cs} (x{n_chunks}) TP=2"
            result = run_forward_backward(model, hidden, mask, config, chunk_size=cs, device=self.device)
            m = compare(baseline, result, atol, rtol, label)
            metrics.append(m)

        if rank == 0:
            print_report(metrics, section + " [TP=2]")

        for m in metrics:
            self.assertTrue(
                m["passed"],
                f"FAILED {m['label']}: out={m['max_output_diff']:.2e} "
                f"loss={m['loss_diff']:.2e} grad={m['max_grad_diff']:.2e} (atol={atol})",
            )

    # ------------------------------------------------------------------
    # Flash attention tests for TP=2
    # ------------------------------------------------------------------
    def test_short_seq_flash_tp2(self):
        try:
            from flash_attn import flash_attn_varlen_func  # noqa: F401
        except ImportError:
            self.skipTest("flash_attn not installed")
        self._run_suite(
            64,
            [32, 16],
            torch.bfloat16,
            True,
            False,
            "Flash Attn | bf16 | Short seq=64 | TP=2",
        )

    def test_medium_seq_flash_tp2(self):
        try:
            from flash_attn import flash_attn_varlen_func  # noqa: F401
        except ImportError:
            self.skipTest("flash_attn not installed")
        self._run_suite(
            256,
            [128, 64],
            torch.bfloat16,
            True,
            False,
            "Flash Attn | bf16 | Medium seq=256 | TP=2",
        )

    def test_long_seq_flash_tp2(self):
        try:
            from flash_attn import flash_attn_varlen_func  # noqa: F401
        except ImportError:
            self.skipTest("flash_attn not installed")
        self._run_suite(
            1024,
            [512, 256],
            torch.bfloat16,
            True,
            False,
            "Flash Attn | bf16 | Long seq=1024 | TP=2",
        )


if __name__ == "__main__":
    # Parse arguments to determine which test suite to run
    parser = argparse.ArgumentParser()
    parser.add_argument("test_class", nargs="?", default="TestChunkedValidation",
                        help="Test class to run (TestChunkedValidation or TestChunkedValidationTP2)")
    args, unknown = parser.parse_known_args()

    # Restore sys.argv for unittest
    sys.argv = [sys.argv[0]] + unknown

    if args.test_class == "TestChunkedValidationTP2":
        suite = unittest.TestLoader().loadTestsFromTestCase(TestChunkedValidationTP2)
    else:
        suite = unittest.TestLoader().loadTestsFromTestCase(TestChunkedValidation)

    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite)
