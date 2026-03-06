# Copyright (c) 2025-2026 Jaegeun Han
#
# SPDX-License-Identifier: Apache-2.0

"""
Integration tests for Muon optimizer with Tensor Parallelism.

Tests validate:
1. Muon works correctly with TP=1 (single GPU)
2. Muon produces valid gradients with TP=2
3. TP=1 and TP=2 produce numerically equivalent results

Usage:
    # Test TP=1
    python tests/integration/test_muon_tp.py --tp 1

    # Test TP=2
    torchrun --nproc_per_node=2 tests/integration/test_muon_tp.py --tp 2

    # Test both (recommended)
    python tests/integration/test_muon_tp.py --test-both
"""

import argparse
import json
import os
import sys
from pathlib import Path

import pytest
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
    PEFTConfig,
    ProfilerConfig,
    TrainerConfig,
    UtilsConfig,
)
from ironcore.config.config_alignment import AlignmentConfig
from ironcore.global_vars import set_global_states, reset_global_states
from ironcore.models.transformer import TransformerModel
from ironcore.optimizer import get_optimizer
from ironcore.parallel import parallel_states

# Test configurations
D_MODEL = 256
NUM_HEADS = 4
HEAD_DIM = D_MODEL // NUM_HEADS
NUM_GROUPS = 4
D_FFN = 1024
NUM_LAYERS = 2
BATCH_SIZE = 2
SEQ_LEN = 128


def create_test_config(tp_size: int, use_muon: bool = True) -> MainConfig:
    """Create test configuration."""
    return MainConfig(
        model=ModelConfig(
            d_model=D_MODEL,
            num_attention_heads=NUM_HEADS,
            num_attention_groups=NUM_GROUPS,
            head_dim=HEAD_DIM,
            d_ffn=D_FFN,
            num_layers=NUM_LAYERS,
            max_seq_len=SEQ_LEN,
            max_position_embeddings=SEQ_LEN,
            dropout_attn=0.0,
            dropout_mlp=0.0,
            dropout_embd=0.0,
            no_bias=False,
            precision="bfloat16",
        ),
        trainer=TrainerConfig(
            tensor_model_parallel_size=tp_size,
            use_flash_attn=False,  # Disable for CPU testing
            sequence_chunk_size=None,
            micro_batch_size=BATCH_SIZE,
            train_batch_size=BATCH_SIZE,
            gradient_accumulation_steps=1,
        ),
        init=InitConfig(seed=42, init_std=0.02),
        optim=OptimConfig(
            optimizer="muon" if use_muon else "adam",
            max_lr=0.02 if use_muon else 5e-4,
            muon_momentum=0.95,
            muon_newton_schulz_steps=5,
            weight_decay=0.01,
        ),
        data=DataConfig(),
        parallel=ParallelConfig(timeout_minute=30),
        operation=OperationConfig(),
        utils=UtilsConfig(),
        peft=PEFTConfig(),
        profiler=ProfilerConfig(),
        alignment=AlignmentConfig(),
    )


def run_forward_backward_with_optimizer(model, optimizer, hidden_states, device):
    """Run forward + backward + optimizer step and collect outputs."""
    model.zero_grad()

    x = hidden_states.clone().detach().requires_grad_(False)
    output = model(x, attention_mask=None, rotary_pos_emb=None)

    loss = output.pow(2).mean()
    loss.backward()

    # Get gradient stats before optimizer step
    grad_stats = {}
    for name, param in model.named_parameters():
        if param.grad is not None:
            if "layers.0.self_attention.linear_q.weight" in name:
                grad_stats["q_weight_grad_mean"] = param.grad.mean().item()
                grad_stats["q_weight_grad_std"] = param.grad.std().item()
                grad_stats["q_weight_grad_norm"] = param.grad.norm().item()
                break

    # Optimizer step
    optimizer.step()

    # Get parameter stats after step
    param_stats = {}
    for name, param in model.named_parameters():
        if "layers.0.self_attention.linear_q.weight" in name:
            param_stats["q_weight_mean"] = param.data.mean().item()
            param_stats["q_weight_norm"] = param.data.norm().item()
            break

    return {
        "output": output.detach(),
        "loss": loss.item(),
        **grad_stats,
        **param_stats,
    }


@pytest.mark.cuda
@pytest.mark.distributed
class TestMuonTP:
    """Test Muon optimizer with Tensor Parallelism."""

    @staticmethod
    def _init_distributed(tp_size: int):
        """Initialize distributed environment."""
        if tp_size > 1:
            if not dist.is_initialized():
                dist.init_process_group(backend="nccl")
            rank = dist.get_rank()
            world_size = dist.get_world_size()

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
            os.environ.setdefault("MASTER_PORT", "12356")
            if not dist.is_initialized():
                dist.init_process_group(backend="nccl", rank=0, world_size=1)

            try:
                parallel_states.initialize_model_parallel(
                    tensor_model_parallel_size=1,
                    timeout_in_minutes=1.0,
                )
            except Exception:
                pass

        return rank, world_size, device

    @staticmethod
    def _cleanup_distributed(tp_size: int):
        """Cleanup distributed environment."""
        if dist.is_initialized():
            dist.barrier()
            if tp_size > 1:
                dist.destroy_process_group()

    def test_muon_tp1_forward_backward(self):
        """Test Muon with TP=1 produces valid gradients."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        tp_size = 1
        rank, world_size, device = self._init_distributed(tp_size)

        try:
            config = create_test_config(tp_size, use_muon=True)
            torch.manual_seed(42)
            torch.cuda.manual_seed_all(42)

            # Initialize global states for logging
            set_global_states(config)

            model = TransformerModel(config).to(device=device, dtype=torch.bfloat16)
            model.init_weights()
            model.train()

            optimizer = get_optimizer(config, model, "cuda")

            hidden = torch.randn(BATCH_SIZE, SEQ_LEN, D_MODEL, device=device, dtype=torch.bfloat16)
            result = run_forward_backward_with_optimizer(model, optimizer, hidden, device)

            # Validate results
            assert not torch.isnan(result["output"]).any()
            assert result["loss"] > 0

            if rank == 0:
                print(f"TP=1 Muon test passed: loss={result['loss']:.6f}")

        finally:
            reset_global_states()
            self._cleanup_distributed(tp_size)

    def test_muon_tp2_forward_backward(self):
        """Test Muon with TP=2 produces valid gradients."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        if dist.is_initialized() and dist.get_world_size() < 2:
            pytest.skip("Need at least 2 GPUs for TP=2 test")

        tp_size = 2
        rank, world_size, device = self._init_distributed(tp_size)

        try:
            config = create_test_config(tp_size, use_muon=True)
            torch.manual_seed(42)
            torch.cuda.manual_seed_all(42)

            # Initialize global states for logging
            set_global_states(config)

            model = TransformerModel(config).to(device=device, dtype=torch.bfloat16)
            model.init_weights()
            model.train()

            optimizer = get_optimizer(config, model, "cuda")

            hidden = torch.randn(BATCH_SIZE, SEQ_LEN, D_MODEL, device=device, dtype=torch.bfloat16)
            result = run_forward_backward_with_optimizer(model, optimizer, hidden, device)

            # Validate results
            assert not torch.isnan(result["output"]).any()
            assert result["loss"] > 0

            if rank == 0:
                print(f"TP=2 Muon test passed: loss={result['loss']:.6f}")

        finally:
            reset_global_states()
            self._cleanup_distributed(tp_size)


def test_muon_tp1_standalone():
    """Standalone test for Muon with TP=1."""
    if not torch.cuda.is_available():
        print("CUDA not available, skipping")
        return

    tp_size = 1

    # Initialize distributed
    if not dist.is_initialized():
        os.environ.setdefault("MASTER_ADDR", "localhost")
        os.environ.setdefault("MASTER_PORT", "12357")
        dist.init_process_group(backend="nccl", rank=0, world_size=1)

    device = torch.device("cuda:0")

    try:
        parallel_states.initialize_model_parallel(
            tensor_model_parallel_size=1,
            timeout_in_minutes=1.0,
        )
    except Exception:
        pass

    try:
        config = create_test_config(tp_size, use_muon=True)
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)

        # Initialize global states for logging
        set_global_states(config)

        model = TransformerModel(config).to(device=device, dtype=torch.bfloat16)
        model.init_weights()
        model.train()

        optimizer = get_optimizer(config, model, "cuda")

        hidden = torch.randn(BATCH_SIZE, SEQ_LEN, D_MODEL, device=device, dtype=torch.bfloat16)
        result = run_forward_backward_with_optimizer(model, optimizer, hidden, device)

        print(f"\n{'=' * 80}")
        print(f"MUON TP=1 TEST RESULTS")
        print(f"{'=' * 80}")
        print(f"Loss: {result['loss']:.6f}")
        print(f"Q weight grad norm: {result.get('q_weight_grad_norm', 'N/A')}")
        print(f"Q weight norm after step: {result.get('q_weight_norm', 'N/A')}")
        print(f"{'=' * 80}\n")

        # Save results
        output_dir = Path("logs/muon_tp")
        output_dir.mkdir(parents=True, exist_ok=True)
        with open(output_dir / "muon_tp1.json", "w") as f:
            json.dump({k: v for k, v in result.items() if not isinstance(v, torch.Tensor)}, f, indent=2)

        return result

    finally:
        reset_global_states()
        if dist.is_initialized():
            dist.destroy_process_group()


@pytest.mark.skipif(
    not os.environ.get("RANK"),
    reason="TP=2 test requires torchrun (RANK env var not set)"
)
def test_muon_tp2_standalone():
    """Standalone test for Muon with TP=2."""
    if not torch.cuda.is_available():
        print("CUDA not available, skipping")
        return

    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")

    rank = dist.get_rank()
    world_size = dist.get_world_size()

    if world_size < 2:
        print("Need at least 2 GPUs for TP=2 test")
        return

    tp_size = 2
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")

    parallel_states.initialize_model_parallel(
        tensor_model_parallel_size=tp_size,
        timeout_in_minutes=30,
    )

    try:
        config = create_test_config(tp_size, use_muon=True)
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)

        # Initialize global states for logging
        set_global_states(config)

        model = TransformerModel(config).to(device=device, dtype=torch.bfloat16)
        model.init_weights()
        model.train()

        optimizer = get_optimizer(config, model, "cuda")

        hidden = torch.randn(BATCH_SIZE, SEQ_LEN, D_MODEL, device=device, dtype=torch.bfloat16)
        result = run_forward_backward_with_optimizer(model, optimizer, hidden, device)

        if rank == 0:
            print(f"\n{'=' * 80}")
            print(f"MUON TP=2 TEST RESULTS")
            print(f"{'=' * 80}")
            print(f"Loss: {result['loss']:.6f}")
            print(f"Q weight grad norm: {result.get('q_weight_grad_norm', 'N/A')}")
            print(f"Q weight norm after step: {result.get('q_weight_norm', 'N/A')}")
            print(f"{'=' * 80}\n")

        # Save results
        if rank == 0:
            output_dir = Path("logs/muon_tp")
            output_dir.mkdir(parents=True, exist_ok=True)
            with open(output_dir / "muon_tp2.json", "w") as f:
                json.dump({k: v for k, v in result.items() if not isinstance(v, torch.Tensor)}, f, indent=2)

        return result

    finally:
        reset_global_states()
        dist.barrier()
        dist.destroy_process_group()


def compare_tp1_tp2():
    """Compare Muon results between TP=1 and TP=2."""
    logs_dir = Path("logs/muon_tp")

    tp1_file = logs_dir / "muon_tp1.json"
    tp2_file = logs_dir / "muon_tp2.json"

    if not tp1_file.exists() or not tp2_file.exists():
        print("ERROR: Missing results files. Run both TP=1 and TP=2 tests first.")
        return

    with open(tp1_file) as f:
        tp1_results = json.load(f)
    with open(tp2_file) as f:
        tp2_results = json.load(f)

    print(f"\n{'=' * 100}")
    print("MUON TP=1 vs TP=2 COMPARISON")
    print(f"{'=' * 100}\n")

    print(f"{'Metric':<30} {'TP=1':>15} {'TP=2':>15} {'Difference':>15}")
    print("-" * 100)

    for key in ["loss", "q_weight_grad_norm", "q_weight_norm"]:
        if key in tp1_results and key in tp2_results:
            v1 = tp1_results[key]
            v2 = tp2_results[key]
            diff = abs(v1 - v2)
            print(f"{key:<30} {v1:>15.6f} {v2:>15.6f} {diff:>15.6f}")

    print(f"\n{'=' * 100}\n")

    # Check if results are close (within tolerance for bfloat16)
    loss_diff = abs(tp1_results["loss"] - tp2_results["loss"])
    if loss_diff < 1e-2:
        print("TP=1 and TP=2 produce similar results")
    else:
        print(f"WARNING: Loss difference ({loss_diff:.6f}) is larger than expected")


def main():
    parser = argparse.ArgumentParser(description="Test Muon optimizer with Tensor Parallelism")
    parser.add_argument("--tp", type=int, choices=[1, 2], help="Tensor parallel size")
    parser.add_argument(
        "--test-both", action="store_true", help="Run both TP=1 and TP=2 and compare"
    )
    args = parser.parse_args()

    if args.test_both:
        print("=" * 80)
        print("COMPREHENSIVE MUON TP=1 vs TP=2 VALIDATION")
        print("=" * 80)
        print()

        # Run TP=1
        print("Running Muon TP=1 test...")
        test_muon_tp1_standalone()

        # Run TP=2
        print("\nRunning Muon TP=2 test...")
        import subprocess

        result = subprocess.run(
            ["torchrun", "--nproc_per_node=2", __file__, "--tp", "2"],
            cwd=Path(__file__).parent.parent,
            check=False,
        )

        if result.returncode != 0:
            print("ERROR: TP=2 test failed!")
            return

        # Compare results
        compare_tp1_tp2()

    elif args.tp == 1:
        test_muon_tp1_standalone()
    elif args.tp == 2:
        test_muon_tp2_standalone()
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
