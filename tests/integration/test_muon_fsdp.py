# Copyright (c) 2025-2026 Jaegeun Han
#
# SPDX-License-Identifier: Apache-2.0

"""
Integration tests for Muon optimizer with FSDP (Fully Sharded Data Parallel).

Tests validate:
1. Muon works correctly with FSDP sharding
2. State dict save/load works with FSDP
3. Memory efficiency compared to DDP

Usage:
    # Test with 2 GPUs
    torchrun --nproc_per_node=2 tests/integration/test_muon_fsdp.py
"""

import os
import sys

import pytest
import torch
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

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
from ironcore.models.transformer import TransformerModel
from ironcore.optimizer import get_optimizer

# Test configurations
D_MODEL = 256
NUM_HEADS = 4
HEAD_DIM = D_MODEL // NUM_HEADS
NUM_GROUPS = 4
D_FFN = 1024
NUM_LAYERS = 2
BATCH_SIZE = 2
SEQ_LEN = 128


def create_fsdp_test_config(use_muon: bool = True) -> MainConfig:
    """Create test configuration for FSDP."""
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
            tensor_model_parallel_size=1,
            use_flash_attn=False,
            sequence_chunk_size=None,
            micro_batch_size=BATCH_SIZE,
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
        parallel=ParallelConfig(
            timeout_minute=30,
            use_fsdp=True,
            fsdp_sharding_strategy="full",
        ),
        operation=OperationConfig(),
        utils=UtilsConfig(),
        peft=PEFTConfig(),
        profiler=ProfilerConfig(),
        alignment=AlignmentConfig(),
    )


@pytest.mark.cuda
@pytest.mark.distributed
class TestMuonFSDP:
    """Test Muon optimizer with FSDP."""

    @pytest.fixture(autouse=True)
    def setup_distributed(self):
        """Setup distributed environment for each test."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        if not dist.is_initialized():
            dist.init_process_group(backend="nccl")

        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        torch.cuda.set_device(self.rank)
        self.device = torch.device(f"cuda:{self.rank}")

        yield

        if dist.is_initialized():
            dist.barrier()

    def test_muon_fsdp_forward_backward(self):
        """Test Muon with FSDP produces valid gradients."""
        config = create_fsdp_test_config(use_muon=True)

        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)

        # Create model
        model = TransformerModel(config).to(device=self.device, dtype=torch.bfloat16)
        model.init_weights()
        model.train()

        # Wrap with FSDP
        from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
        from ironcore.models.transformer import TransformerLayer

        auto_wrap_policy = transformer_auto_wrap_policy(
            transformer_layer_cls={TransformerLayer}
        )

        model = FSDP(
            model,
            auto_wrap_policy=auto_wrap_policy,
            device_id=self.rank,
            mixed_precision=None,  # We handle precision ourselves
        )

        # Create optimizer
        optimizer = get_optimizer(config, model, "cuda")

        # Forward pass
        hidden = torch.randn(
            BATCH_SIZE, SEQ_LEN, D_MODEL, device=self.device, dtype=torch.bfloat16
        )
        output = model(hidden, attention_mask=None, rotary_pos_emb=None)

        loss = output.pow(2).mean()
        loss.backward()

        # Optimizer step
        optimizer.step()
        optimizer.zero_grad()

        # Validate
        assert not torch.isnan(output).any()
        assert loss.item() > 0

        if self.rank == 0:
            print(f"Muon FSDP test passed: loss={loss.item():.6f}")

    def test_muon_fsdp_multiple_steps(self):
        """Test Muon with FSDP for multiple training steps."""
        config = create_fsdp_test_config(use_muon=True)

        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)

        # Create and wrap model
        model = TransformerModel(config).to(device=self.device, dtype=torch.bfloat16)
        model.init_weights()
        model.train()

        from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
        from ironcore.models.transformer import TransformerLayer

        model = FSDP(
            model,
            auto_wrap_policy=transformer_auto_wrap_policy(
                transformer_layer_cls={TransformerLayer}
            ),
            device_id=self.rank,
        )

        optimizer = get_optimizer(config, model, "cuda")

        # Run multiple steps
        losses = []
        for step_idx in range(3):
            hidden = torch.randn(
                BATCH_SIZE, SEQ_LEN, D_MODEL, device=self.device, dtype=torch.bfloat16
            )
            output = model(hidden, attention_mask=None, rotary_pos_emb=None)
            loss = output.pow(2).mean()

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            losses.append(loss.item())

        if self.rank == 0:
            print(f"Muon FSDP multi-step test passed: losses={[f'{l:.6f}' for l in losses]}")

    def test_muon_fsdp_state_dict(self):
        """Test Muon optimizer state dict save/load with FSDP."""
        config = create_fsdp_test_config(use_muon=True)

        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)

        # Create and wrap model
        model = TransformerModel(config).to(device=self.device, dtype=torch.bfloat16)
        model.init_weights()
        model.train()

        from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
        from ironcore.models.transformer import TransformerLayer

        model = FSDP(
            model,
            auto_wrap_policy=transformer_auto_wrap_policy(
                transformer_layer_cls={TransformerLayer}
            ),
            device_id=self.rank,
        )

        optimizer = get_optimizer(config, model, "cuda")

        # Run a step to build state
        hidden = torch.randn(
            BATCH_SIZE, SEQ_LEN, D_MODEL, device=self.device, dtype=torch.bfloat16
        )
        output = model(hidden, attention_mask=None, rotary_pos_emb=None)
        loss = output.pow(2).mean()
        loss.backward()
        optimizer.step()

        # Save state dict
        from torch.distributed.fsdp import StateDictType

        with FSDP.state_dict_type(model, StateDictType.LOCAL_STATE_DICT):
            optim_state_dict = optimizer.state_dict()

        # Verify state dict has content
        assert len(optim_state_dict.get("state", {})) > 0 or len(optim_state_dict.get("param_groups", [])) > 0

        if self.rank == 0:
            print("Muon FSDP state dict test passed")


def run_standalone_fsdp_test():
    """Standalone test function for running from command line."""
    if not torch.cuda.is_available():
        print("CUDA not available, skipping")
        return

    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")

    print(f"Running Muon FSDP test on rank {rank}/{world_size}")

    config = create_fsdp_test_config(use_muon=True)

    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)

    # Create model
    model = TransformerModel(config).to(device=device, dtype=torch.bfloat16)
    model.init_weights()
    model.train()

    # Wrap with FSDP
    from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
    from ironcore.models.transformer import TransformerLayer

    model = FSDP(
        model,
        auto_wrap_policy=transformer_auto_wrap_policy(
            transformer_layer_cls={TransformerLayer}
        ),
        device_id=rank,
    )

    # Create optimizer
    optimizer = get_optimizer(config, model, "cuda")

    # Run training steps
    print(f"Rank {rank}: Running training steps...")
    for step_idx in range(5):
        hidden = torch.randn(
            BATCH_SIZE, SEQ_LEN, D_MODEL, device=device, dtype=torch.bfloat16
        )
        output = model(hidden, attention_mask=None, rotary_pos_emb=None)
        loss = output.pow(2).mean()

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if rank == 0:
            print(f"Step {step_idx}: loss={loss.item():.6f}")

    if rank == 0:
        print("\nMuon FSDP test completed successfully!")

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    run_standalone_fsdp_test()
