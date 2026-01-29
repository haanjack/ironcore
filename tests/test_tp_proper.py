#!/usr/bin/env python3
"""
Proper TP=1 vs TP=2 evaluation with SAME weights.

Correct methodology:
1. Create and save model with TP=1
2. Load the saved weights into TP=2 (with proper sharding)
3. Run same input on both
4. Compare outputs (should be identical)
5. Measure memory usage

Run with:
- TP=1: python tests/test_tp_proper.py --mode save_weights
- TP=2: torchrun --nproc_per_node=2 tests/test_tp_proper.py --mode load_and_compare
"""
import os
import sys
import torch
import torch.distributed as dist
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ironcore.config import (
    MainConfig, ModelConfig, TrainerConfig, InitConfig, OptimConfig,
    DataConfig, ParallelConfig, OperationConfig, UtilsConfig, load_trainer_config
)
from ironcore.global_vars import set_global_states
from ironcore.language_model import LanguageModel
from ironcore.parallel import parallel_states
from ironcore.parallel.parallel import initialize_process


def create_model_config(tp_size=1, use_flash_attn=False):
    """Create a small model config for testing."""
    model_config = ModelConfig(
        d_model=256,  # Small model for testing
        num_attention_heads=8,
        num_attention_groups=8,
        head_dim=32,
        max_seq_len=64,
        max_position_embeddings=64,
        dropout_attn=0.0,
        no_bias=False,
    )
    # Add name attribute for model provider selection
    model_config.name = "gpt2"

    trainer_config = TrainerConfig(
        tensor_model_parallel_size=tp_size,
        use_flash_attn=use_flash_attn,
    )

    init_config = InitConfig(seed=42, init_std=0.02)
    optim_config = OptimConfig(max_lr=1e-3, weight_decay=0.01)
    data_config = DataConfig()
    parallel_config = ParallelConfig()
    operation_config = OperationConfig(train_steps=100)
    utils_config = UtilsConfig()

    return MainConfig(
        model=model_config, trainer=trainer_config, init=init_config,
        optim=optim_config, data=data_config, parallel=parallel_config,
        operation=operation_config, utils=utils_config,
    )


def save_tp1_model():
    """Create and save TP=1 model."""
    print("\n" + "="*70)
    print("MODE: Save TP=1 Model Weights")
    print("="*70)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Create config
    config = create_model_config(tp_size=1)

    # Set global states - this will build tokenizer etc.
    from ironcore.global_vars import set_global_states
    set_global_states(config)

    # Initialize TP=1
    parallel_states.initialize_model_parallel(tensor_model_parallel_size=1, timeout_in_minutes=10.0)

    # Set seed for reproducibility
    import random
    seed = 42
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Create model
    def dummy_loss_fn(loss, mask):
        return (loss * mask).sum() / mask.sum()

    model = LanguageModel(config, dummy_loss_fn).to(device)

    # Create sample input
    batch_size = 2
    seq_len = 32
    vocab_size = config.data.vocab_size

    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    labels = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)

    # Forward pass to get activations
    print("Running forward pass...")
    with torch.no_grad():
        output = model(input_ids, labels)

    # Save model
    os.makedirs("checkpoints/tp1", exist_ok=True)
    checkpoint_path = "checkpoints/tp1/model.pt"
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config,
        'input_ids': input_ids,
        'labels': labels,
        'output': output,
    }, checkpoint_path)

    print(f"Model saved to {checkpoint_path}")

    # Measure memory
    mem_peak = torch.cuda.max_memory_allocated() / 1024**2
    param_count = sum(p.numel() for p in model.parameters())

    print(f"\nTP=1 Statistics:")
    print(f"  Parameters: {param_count:,}")
    print(f"  Peak memory: {mem_peak:.2f} MB")
    print(f"  Output norm: {output.norm().item():.6f}")
    print(f"  Output mean: {output.mean().item():.6f}")
    print(f"  Output std: {output.std().item():.6f}")

    return checkpoint_path, output


def load_and_compare_tp2():
    """Load TP=1 weights into TP=2 and compare."""
    print("\n" + "="*70)
    print("MODE: Load TP=1 Weights into TP=2 and Compare")
    print("="*70)

    # Get rank info
    if 'RANK' not in os.environ:
        print("ERROR: Must run with torchrun for TP=2")
        print("Usage: torchrun --nproc_per_node=2 python tests/test_tp_proper.py --mode load_and_compare")
        sys.exit(1)

    rank = int(os.environ['RANK'])
    local_rank = int(os.environ['LOCAL_RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    device = torch.device(f'cuda:{local_rank}')

    # Initialize distributed
    dist.init_process_group(backend='nccl')
    torch.cuda.set_device(local_rank)

    # Load checkpoint
    checkpoint_path = "checkpoints/tp1/model.pt"
    if not os.path.exists(checkpoint_path):
        print(f"ERROR: Checkpoint not found: {checkpoint_path}")
        print("Run first with: python tests/test_tp_proper.py --mode save_weights")
        sys.exit(1)

    if rank == 0:
        print(f"Loading checkpoint from {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Create TP=2 config
    config = create_model_config(tp_size=2, use_flash_attn=False)

    # Set global states for distributed
    from ironcore.global_vars import set_global_states
    set_global_states(config)

    # Initialize TP=2
    parallel_states.initialize_model_parallel(tensor_model_parallel_size=2, timeout_in_minutes=10.0)

    # Set same seed
    import random
    seed = 42
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Create model
    def dummy_loss_fn(loss, mask):
        return (loss * mask).sum() / mask.sum()

    model = LanguageModel(config, dummy_loss_fn).to(device)

    # Load TP=1 weights into TP=2 model
    if rank == 0:
        print("Loading TP=1 weights into TP=2 model...")

    model.load_state_dict(checkpoint['model_state_dict'])

    # Get the same input
    input_ids = checkpoint['input_ids'].to(device)
    labels = checkpoint['labels'].to(device)

    # Verify input is same across ranks
    input_sum = input_ids.sum().item()
    input_sums = [None] * world_size
    dist.all_gather_object(input_sums, input_sum)

    if rank == 0:
        if len(set(input_sums)) == 1:
            print(f"✓ Input is identical across all ranks (sum={input_sums[0]})")
        else:
            print(f"✗ ERROR: Input differs across ranks: {input_sums}")

    # Measure memory before forward
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()

    # Forward pass
    print(f"[Rank {rank}] Running forward pass...")
    with torch.no_grad():
        output_tp2 = model(input_ids, labels)

    # Measure memory
    mem_peak = torch.cuda.max_memory_allocated() / 1024**2
    local_params = sum(p.numel() for p in model.parameters())

    # Get original TP=1 output
    output_tp1 = checkpoint['output'].to(device)

    # Compare outputs
    output_diff = (output_tp2 - output_tp1).abs().max().item()
    output_norm_diff = (output_tp2.norm() - output_tp1.norm()).abs().item()

    print(f"[Rank {rank}] TP=2 Statistics:")
    print(f"  Local params: {local_params:,}")
    print(f"  Peak memory: {mem_peak:.2f} MB")
    print(f"  Output norm: {output_tp2.norm().item():.6f}")

    # Gather results
    all_mem_peak = [None] * world_size
    all_output_diff = [None] * world_size
    all_output_norm_diff = [None] * world_size

    dist.all_gather(all_mem_peak, torch.tensor(mem_peak, device=device))
    dist.all_gather(all_output_diff, torch.tensor(output_diff, device=device))
    dist.all_gather(all_output_norm_diff, torch.tensor(output_norm_diff, device=device))

    if rank == 0:
        print(f"\n{'='*70}")
        print(f"TP=1 vs TP=2 COMPARISON (SAME WEIGHTS)")
        print(f"{'='*70}")

        print(f"\nOutput Difference:")
        print(f"  Max absolute difference: {all_output_diff[0].item():.2e}")
        print(f"  Norm difference: {all_output_norm_diff[0].item():.2e}")

        tol = 1e-5
        if all_output_diff[0].item() < tol:
            print(f"  ✓ Outputs are IDENTICAL (diff < {tol})")
        else:
            print(f"  ✗ Outputs DIFFER (diff = {all_output_diff[0].item():.2e})")

        print(f"\nMemory Usage:")
        for i in range(world_size):
            print(f"  Rank {i} peak memory: {all_mem_peak[i].item():.2f} MB")

        avg_mem = sum(m.item() for m in all_mem_peak) / world_size
        print(f"  Average per GPU: {avg_mem:.2f} MB")
        print(f"  Total (2 GPUs): {sum(m.item() for m in all_mem_peak):.2f} MB")

    dist.barrier()
    dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, required=True,
                       choices=['save_weights', 'load_and_compare'],
                       help='Mode: save_weights (TP=1) or load_and_compare (TP=2)')
    args = parser.parse_args()

    if args.mode == 'save_weights':
        save_tp1_model()
    elif args.mode == 'load_and_compare':
        load_and_compare_tp2()


if __name__ == "__main__":
    main()
