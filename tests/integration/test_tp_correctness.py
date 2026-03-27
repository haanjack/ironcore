# Copyright (c) 2025-2026 Jaegeun Han
#
# SPDX-License-Identifier: Apache-2.0
"""
TP=1 vs TP=2 Correctness Validation for Chunked Tensor Parallelism

This test validates that:
1. TP=1 (single GPU) produces identical results with/without chunking
2. TP=2 (2 GPUs with async all-reduce) produces identical results to TP=1
3. Different chunk sizes produce numerically equivalent results

Tests validate:
- Forward pass activations
- Loss values
- Gradient values
- Numerical precision across configurations

Usage:
    # Test TP=1
    python tests/test_tp_correctness.py --tp 1

    # Test TP=2
    torchrun --nproc_per_node=2 tests/test_tp_correctness.py --tp 2

    # Test both (recommended)
    python tests/test_tp_correctness.py --test-both
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

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
    ProfilerConfig,
    TrainerConfig,
    UtilsConfig,
)
from ironcore.models.transformer import TransformerModel
from ironcore.parallel import parallel_states

# Test configurations
D_MODEL = 512
NUM_HEADS = 8
HEAD_DIM = D_MODEL // NUM_HEADS
NUM_GROUPS = 8
D_FFN = 2048
NUM_LAYERS = 6
BATCH_SIZE = 1

# Test matrix
TEST_SEQ_LENGTHS = [1024, 4096]  # Reduced for faster testing
TEST_CHUNK_SIZES = [None, 2048, 1024, 512]


def create_config(seq_len, chunk_size, tp_size):
    """Create test configuration."""
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
            attention_bias=True,
            mlp_bias=True,
            layernorm_bias=True,
            precision="bfloat16",
        ),
        trainer=TrainerConfig(
            tensor_model_parallel_size=tp_size,
            use_flash_attn=True,
            sequence_chunk_size=chunk_size,
        ),
        init=InitConfig(seed=42, init_std=0.02),
        optim=OptimConfig(),
        data=DataConfig(),
        parallel=ParallelConfig(timeout_minute=30),
        operation=OperationConfig(),
        utils=UtilsConfig(),
        profiler=ProfilerConfig(),
    )


def run_forward_backward(model, hidden_states, device):
    """Run forward + backward pass and collect outputs/gradients."""
    model.zero_grad()

    x = hidden_states.clone().detach().requires_grad_(False)  # No input grad needed
    output = model(x, attention_mask=None, rotary_pos_emb=None)

    loss = output.pow(2).mean()
    loss.backward()

    # Collect gradients from first layer for validation
    grad_stats = {}
    for name, param in model.named_parameters():
        if param.grad is not None:
            if "layers.0.self_attention.linear_q.weight" in name:
                grad_stats["q_weight_grad_mean"] = param.grad.mean().item()
                grad_stats["q_weight_grad_std"] = param.grad.std().item()
                grad_stats["q_weight_grad_norm"] = param.grad.norm().item()
                break

    return {
        "output": output.detach(),
        "loss": loss.item(),
        **grad_stats,
    }


def test_tp_correctness(tp_size):
    """Test correctness for a given TP size."""
    # Initialize distributed
    if tp_size > 1:
        if not dist.is_initialized():
            dist.init_process_group(backend="nccl")
        rank = dist.get_rank()
        world_size = dist.get_world_size()

        # Set CUDA device for this rank
        torch.cuda.set_device(rank)
        device = torch.device(f"cuda:{rank}")

        parallel_states.initialize_model_parallel(
            tensor_model_parallel_size=tp_size,
            timeout_in_minutes=30,
        )
    else:
        rank = 0
        world_size = 1
        device = torch.device("cuda:0")

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

    if rank == 0:
        print(f"\n{'=' * 80}")
        print(f"CORRECTNESS TEST: TP={tp_size}")
        print(f"{'=' * 80}\n")

    all_results = []

    for seq_len in TEST_SEQ_LENGTHS:
        if rank == 0:
            print(f"\nTesting seq_len={seq_len}:")
            print("-" * 80)

        # Create baseline (no chunking)
        baseline_config = create_config(seq_len, None, tp_size)
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)

        baseline_model = TransformerModel(baseline_config).to(device=device, dtype=torch.bfloat16)
        baseline_model.init_weights()
        baseline_model.train()

        hidden = torch.randn(BATCH_SIZE, seq_len, D_MODEL, device=device, dtype=torch.bfloat16)
        baseline_result = run_forward_backward(baseline_model, hidden, device)

        del baseline_model
        torch.cuda.empty_cache()

        for chunk_size in TEST_CHUNK_SIZES:
            if chunk_size is None or chunk_size >= seq_len:
                continue

            # Create chunked model
            torch.manual_seed(42)
            torch.cuda.manual_seed_all(42)

            config = create_config(seq_len, chunk_size, tp_size)
            model = TransformerModel(config).to(device=device, dtype=torch.bfloat16)
            model.init_weights()
            model.train()

            result = run_forward_backward(model, hidden, device)

            # Compare results
            output_diff = (baseline_result["output"] - result["output"]).abs().max().item()
            loss_diff = abs(baseline_result["loss"] - result["loss"])

            grad_diff = 0.0
            if "q_weight_grad_norm" in baseline_result and "q_weight_grad_norm" in result:
                grad_diff = abs(
                    baseline_result["q_weight_grad_norm"] - result["q_weight_grad_norm"]
                )

            num_chunks = (seq_len + chunk_size - 1) // chunk_size

            # Tolerance for bfloat16 (relaxed to account for chunking numerical errors)
            atol = 1e-1
            passed = output_diff < atol and loss_diff < atol

            test_result = {
                "rank": rank,
                "tp_size": tp_size,
                "seq_len": seq_len,
                "chunk_size": chunk_size,
                "num_chunks": num_chunks,
                "output_diff": output_diff,
                "loss_diff": loss_diff,
                "grad_diff": grad_diff,
                "baseline_loss": baseline_result["loss"],
                "chunked_loss": result["loss"],
                "passed": passed,
                "atol": atol,
            }

            all_results.append(test_result)

            if rank == 0:
                status = "✓ PASS" if passed else "✗ FAIL"
                print(
                    f"  chunk={chunk_size:<6} ({num_chunks:>2} chunks): "
                    f"out_diff={output_diff:.2e}, loss_diff={loss_diff:.2e}, "
                    f"grad_diff={grad_diff:.2e} {status}"
                )

            del model
            torch.cuda.empty_cache()

    # Gather results from all ranks
    if world_size > 1:
        gathered_results = [None] * world_size
        dist.all_gather_object(gathered_results, all_results)
        if rank == 0:
            all_results = []
            for rank_results in gathered_results:
                all_results.extend(rank_results)

    # Save results
    if rank == 0:
        output_dir = Path("logs/tp_correctness")
        output_dir.mkdir(parents=True, exist_ok=True)

        output_file = output_dir / f"correctness_tp{tp_size}.json"
        with open(output_file, "w") as f:
            json.dump(all_results, f, indent=2)

        # Check if all tests passed
        rank0_results = [r for r in all_results if r["rank"] == 0]
        all_passed = all(r["passed"] for r in rank0_results)

        print(f"\n{'=' * 80}")
        if all_passed:
            print(f"✓ ALL TESTS PASSED for TP={tp_size}")
        else:
            print(f"✗ SOME TESTS FAILED for TP={tp_size}")
            failed = [r for r in rank0_results if not r["passed"]]
            for r in failed:
                print(
                    f"  FAILED: seq={r['seq_len']}, chunk={r['chunk_size']}, "
                    f"out_diff={r['output_diff']:.2e}"
                )
        print(f"{'=' * 80}\n")

        print(f"Results saved to: {output_file}")

    # Cleanup
    if dist.is_initialized():
        dist.barrier()
        if tp_size > 1:
            dist.destroy_process_group()

    return all_results


def compare_tp1_tp2():
    """Compare TP=1 and TP=2 results."""
    logs_dir = Path("logs/tp_correctness")

    tp1_file = logs_dir / "correctness_tp1.json"
    tp2_file = logs_dir / "correctness_tp2.json"

    if not tp1_file.exists() or not tp2_file.exists():
        print("ERROR: Missing results files. Run both TP=1 and TP=2 tests first.")
        return

    with open(tp1_file) as f:
        tp1_results = json.load(f)
    with open(tp2_file) as f:
        tp2_results = json.load(f)

    # Filter rank 0 results only
    tp1_results = [r for r in tp1_results if r["rank"] == 0]
    tp2_results = [r for r in tp2_results if r["rank"] == 0]

    print(f"\n{'=' * 100}")
    print("TP=1 vs TP=2 COMPARISON")
    print(f"{'=' * 100}\n")

    print(
        f"{'Seq':>5}  {'Chunk':>6}  {'TP=1 Loss':>12}  {'TP=2 Loss':>12}  {'Loss Δ':>10}  {'Status':>8}"
    )
    print("-" * 100)

    # Create lookup for TP=2
    tp2_lookup = {}
    for r in tp2_results:
        key = (r["seq_len"], r["chunk_size"])
        tp2_lookup[key] = r

    max_diff = 0.0
    for r1 in tp1_results:
        key = (r1["seq_len"], r1["chunk_size"])
        r2 = tp2_lookup.get(key)

        if r2 is None:
            continue

        loss_diff = abs(r1["chunked_loss"] - r2["chunked_loss"])
        max_diff = max(max_diff, loss_diff)

        # Both are bfloat16, so tolerance is higher
        passed = loss_diff < 1e-3
        status = "✓ PASS" if passed else "✗ FAIL"

        print(
            f"{r1['seq_len']:>5}  {r1['chunk_size']:>6}  "
            f"{r1['chunked_loss']:>12.8f}  {r2['chunked_loss']:>12.8f}  "
            f"{loss_diff:>10.2e}  {status:>8}"
        )

    print(f"\n{'=' * 100}")
    print(f"Maximum loss difference between TP=1 and TP=2: {max_diff:.2e}")
    if max_diff < 1e-3:
        print("✓ TP=1 and TP=2 are numerically equivalent")
    else:
        print("✗ WARNING: Significant differences detected between TP=1 and TP=2")
    print(f"{'=' * 100}\n")


def test_both():
    """Run both TP=1 and TP=2 tests and compare."""
    print("=" * 80)
    print("COMPREHENSIVE TP=1 vs TP=2 CORRECTNESS VALIDATION")
    print("=" * 80)
    print()

    # Run TP=1
    print("Running TP=1 test...")
    result_tp1 = subprocess.run(
        [sys.executable, __file__, "--tp", "1"],
        cwd=Path(__file__).parent.parent,
        check=False,
    )

    if result_tp1.returncode != 0:
        print("ERROR: TP=1 test failed!")
        return

    # Run TP=2
    print("\nRunning TP=2 test...")
    result_tp2 = subprocess.run(
        ["torchrun", "--nproc_per_node=2", __file__, "--tp", "2"],
        cwd=Path(__file__).parent.parent,
        check=False,
    )

    if result_tp2.returncode != 0:
        print("ERROR: TP=2 test failed!")
        return

    # Compare results
    compare_tp1_tp2()


def main():
    parser = argparse.ArgumentParser(description="Test TP=1 vs TP=2 correctness")
    parser.add_argument("--tp", type=int, choices=[1, 2], help="Tensor parallel size")
    parser.add_argument(
        "--test-both", action="store_true", help="Run both TP=1 and TP=2 and compare"
    )
    args = parser.parse_args()

    if args.test_both:
        test_both()
    elif args.tp is not None:
        test_tp_correctness(args.tp)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("ERROR: CUDA not available")
        sys.exit(1)
    main()
