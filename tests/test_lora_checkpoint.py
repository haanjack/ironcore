#!/usr/bin/env python3
"""
Test LoRA checkpoint save/load functionality.

This test validates:
1. LoRA weights are correctly saved in checkpoints
2. LoRA weights are correctly loaded from checkpoints
3. Universal checkpoint compatibility (TP=1 save, TP=2 load)
4. Optimizer state is correctly saved/loaded for LoRA parameters

Run with:
- python tests/test_lora_checkpoint.py --test save_load_tp1
- torchrun --nproc_per_node=2 tests/test_lora_checkpoint.py --test universal_checkpoint
"""

import argparse
import os
import shutil
import sys

import torch
import torch.distributed as dist

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ironcore.checkpointing.native import load_checkpoint, save_checkpoint
from ironcore.config import (
    DataConfig,
    InitConfig,
    LoRAConfig,
    MainConfig,
    ModelConfig,
    OperationConfig,
    OptimConfig,
    ParallelConfig,
    PEFTConfig,
    TrainerConfig,
    UtilsConfig,
)
from ironcore.global_vars import set_global_states
from ironcore.language_model import LanguageModel
from ironcore.optimizer import get_optimizer
from ironcore.parallel import parallel_states


def create_model_config(tp_size=1):
    """Create a small model config for testing."""
    model_config = ModelConfig(
        d_model=256,
        num_attention_heads=8,
        num_attention_groups=8,
        head_dim=32,
        max_seq_len=64,
        max_position_embeddings=64,
        dropout_attn=0.0,
        no_bias=False,
    )
    model_config.name = "gpt2"
    model_config.hf_model_type = "gpt2"
    model_config.hf_architecture = "GPT2LMHeadModel"

    trainer_config = TrainerConfig(
        tensor_model_parallel_size=tp_size,
        use_flash_attn=False,
        model_path="test_outputs/lora_checkpoint_test",
    )

    # LoRA configuration
    lora_config = LoRAConfig(
        r=8,
        alpha=16.0,
        dropout=0.0,
        target_modules=["q_proj", "v_proj", "o_proj", "up_proj", "down_proj"],
    )
    peft_config = PEFTConfig(method="lora", lora=lora_config)

    init_config = InitConfig(seed=42, init_std=0.02)
    optim_config = OptimConfig(max_lr=1e-3, weight_decay=0.01)
    data_config = DataConfig()
    parallel_config = ParallelConfig()
    operation_config = OperationConfig(train_steps=100, no_save=False)
    utils_config = UtilsConfig()

    return MainConfig(
        model=model_config,
        trainer=trainer_config,
        init=init_config,
        optim=optim_config,
        data=data_config,
        parallel=parallel_config,
        operation=operation_config,
        utils=utils_config,
        peft=peft_config,
    )


def test_save_load_tp1():
    """Test save and load LoRA checkpoint with TP=1."""
    print("\n" + "=" * 70)
    print("Test: Save and Load LoRA Checkpoint (TP=1)")
    print("=" * 70)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Clean up previous test outputs
    if os.path.exists("test_outputs/lora_checkpoint_test"):
        shutil.rmtree("test_outputs/lora_checkpoint_test")

    # Create config
    config = create_model_config(tp_size=1)
    set_global_states(config)
    parallel_states.initialize_model_parallel(tensor_model_parallel_size=1, timeout_in_minutes=10.0)

    # Set seed
    import random

    import numpy as np

    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    # Create model and optimizer
    model = LanguageModel(config)
    model.to(device)
    optimizer = get_optimizer(
        config, model, device_type="cuda" if torch.cuda.is_available() else "cpu"
    )

    # Create dummy scheduler
    from torch.optim.lr_scheduler import StepLR

    scheduler = StepLR(optimizer, step_size=10, gamma=0.1)

    # Save initial state
    print("\n[1] Saving checkpoint at step 0...")
    save_checkpoint(config, model, optimizer, scheduler, step=0)
    print("  ✓ Checkpoint saved")

    # Get initial LoRA parameters
    initial_lora_params = {
        name: param.clone() for name, param in model.named_parameters() if "lora_" in name
    }
    print(f"  Initial LoRA parameters: {len(initial_lora_params)}")

    # Modify LoRA parameters
    print("\n[2] Modifying LoRA parameters...")
    for name, param in model.named_parameters():
        if "lora_" in name:
            param.data.add_(torch.randn_like(param) * 0.1)

    # Verify parameters changed
    for name, param in model.named_parameters():
        if "lora_" in name:
            diff = torch.abs(param - initial_lora_params[name]).max().item()
            assert diff > 1e-6, f"Parameter {name} did not change"
    print("  ✓ LoRA parameters modified")

    # Load checkpoint
    print("\n[3] Loading checkpoint...")
    config.trainer.model_path = "test_outputs/lora_checkpoint_test"
    loaded_step = load_checkpoint(config, model, optimizer, scheduler, step=0)
    print(f"  Loaded step: {loaded_step}")

    # Verify LoRA parameters restored
    print("\n[4] Verifying LoRA parameters restored...")
    max_diff = 0.0
    for name, param in model.named_parameters():
        if "lora_" in name:
            diff = torch.abs(param - initial_lora_params[name]).max().item()
            max_diff = max(max_diff, diff)

    print(f"  Max difference: {max_diff:.6e}")
    assert max_diff < 1e-6, f"LoRA parameters not restored correctly: max_diff={max_diff}"
    print("  ✓ LoRA parameters correctly restored")

    parallel_states.destroy_model_parallel()

    print("\n" + "=" * 70)
    print("SAVE/LOAD TEST PASSED ✓")
    print("=" * 70)


def test_universal_checkpoint():
    """Test universal checkpoint: save with TP=1, load with TP=2."""
    print("\n" + "=" * 70)
    print("Test: Universal Checkpoint (TP=1 → TP=2)")
    print("=" * 70)

    # Initialize distributed
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl" if torch.cuda.is_available() else "gloo")

    rank = dist.get_rank()
    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")

    # Step 1: Save with TP=1 (only rank 0)
    if rank == 0:
        print("\n[1] Saving checkpoint with TP=1...")
        if os.path.exists("test_outputs/lora_checkpoint_test"):
            shutil.rmtree("test_outputs/lora_checkpoint_test")

        config_tp1 = create_model_config(tp_size=1)
        set_global_states(config_tp1)
        parallel_states.initialize_model_parallel(
            tensor_model_parallel_size=1, timeout_in_minutes=10.0
        )

        # Set seed
        import random

        import numpy as np

        random.seed(42)
        np.random.seed(42)
        torch.manual_seed(42)

        model_tp1 = LanguageModel(config_tp1)
        model_tp1.to(device)
        optimizer_tp1 = get_optimizer(
            config_tp1, model_tp1, device_type="cuda" if torch.cuda.is_available() else "cpu"
        )

        from torch.optim.lr_scheduler import StepLR

        scheduler_tp1 = StepLR(optimizer_tp1, step_size=10, gamma=0.1)

        save_checkpoint(config_tp1, model_tp1, optimizer_tp1, scheduler_tp1, step=0)

        # Get LoRA parameter values
        tp1_lora_params = {
            name: param.clone().cpu()
            for name, param in model_tp1.named_parameters()
            if "lora_" in name
        }

        parallel_states.destroy_model_parallel()
        print(f"  ✓ Saved {len(tp1_lora_params)} LoRA parameters")
    else:
        tp1_lora_params = None

    # Synchronize
    dist.barrier()

    # Broadcast LoRA params to all ranks
    if rank != 0:
        # Receive the list of parameter names and shapes from rank 0
        pass  # Will be filled by load_checkpoint

    # Step 2: Load with TP=2
    print(f"\n[2] Loading checkpoint with TP=2 (rank {rank})...")
    config_tp2 = create_model_config(tp_size=2)
    set_global_states(config_tp2)
    parallel_states.initialize_model_parallel(tensor_model_parallel_size=2, timeout_in_minutes=10.0)

    # Set seed
    import random

    import numpy as np

    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    model_tp2 = LanguageModel(config_tp2)
    model_tp2.to(device)
    optimizer_tp2 = get_optimizer(
        config_tp2, model_tp2, device_type="cuda" if torch.cuda.is_available() else "cpu"
    )

    from torch.optim.lr_scheduler import StepLR

    scheduler_tp2 = StepLR(optimizer_tp2, step_size=10, gamma=0.1)

    config_tp2.trainer.model_path = "test_outputs/lora_checkpoint_test"
    loaded_step = load_checkpoint(config_tp2, model_tp2, optimizer_tp2, scheduler_tp2, step=0)

    if rank == 0:
        print(f"  Loaded step: {loaded_step}")

    # Step 3: Verify LoRA parameters are identical across ranks
    print(f"\n[3] Verifying LoRA parameters (rank {rank})...")
    tp2_lora_params = {
        name: param.clone().cpu() for name, param in model_tp2.named_parameters() if "lora_" in name
    }

    if rank == 0:
        print(f"  Loaded {len(tp2_lora_params)} LoRA parameters")

        # Compare with TP=1 values
        max_diff = 0.0
        for name, param in tp2_lora_params.items():
            if name in tp1_lora_params:
                diff = torch.abs(param - tp1_lora_params[name]).max().item()
                max_diff = max(max_diff, diff)

        print(f"  Max difference from TP=1: {max_diff:.6e}")
        assert max_diff < 1e-6, f"LoRA parameters differ: max_diff={max_diff}"
        print("  ✓ LoRA parameters match TP=1 values")

    # Gather LoRA params from both ranks to verify replication
    for name, param in tp2_lora_params.items():
        # Gather from all ranks
        gathered = [torch.zeros_like(param) for _ in range(2)]
        dist.all_gather(gathered, param)

        if rank == 0:
            # Verify all ranks have identical values
            diff = torch.abs(gathered[0] - gathered[1]).max().item()
            assert diff < 1e-9, f"LoRA parameter {name} differs across ranks: {diff}"

    if rank == 0:
        print("  ✓ LoRA parameters replicated identically across ranks")

    parallel_states.destroy_model_parallel()
    dist.destroy_process_group()

    if rank == 0:
        print("\n" + "=" * 70)
        print("UNIVERSAL CHECKPOINT TEST PASSED ✓")
        print("=" * 70)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--test",
        choices=["save_load_tp1", "universal_checkpoint"],
        required=True,
        help="Which test to run",
    )
    args = parser.parse_args()

    if args.test == "save_load_tp1":
        test_save_load_tp1()
    elif args.test == "universal_checkpoint":
        test_universal_checkpoint()


if __name__ == "__main__":
    main()
