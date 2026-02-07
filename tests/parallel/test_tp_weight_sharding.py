#!/usr/bin/env python3
"""
Test TP=1 vs TP=2 with SAME weights (properly sharded).
This is the correct approach for validating tensor parallelism.

Methodology:
1. Create and save TP=1 attention weights
2. Load SAME weights into TP=2 with proper sharding
3. Compare outputs (should be identical)
4. Measure memory usage

Run with:
- TP=1: python tests/test_tp_weight_sharding.py --mode save_weights
- TP=2: torchrun --nproc_per_node=2 tests/test_tp_weight_sharding.py --mode load_and_compare
"""
import os
import sys
import torch
import torch.distributed as dist
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ironcore.config import (
    MainConfig, ModelConfig, TrainerConfig, InitConfig, OptimConfig,
    DataConfig, ParallelConfig, OperationConfig, UtilsConfig
)
from ironcore.layers.attention import Attention
from ironcore.parallel import parallel_states


def create_config(tensor_model_parallel_size=1):
    """Create test configuration."""
    model_config = ModelConfig(
        d_model=512,
        num_attention_heads=8,
        num_attention_groups=8,
        head_dim=64,
        max_seq_len=128,
        max_position_embeddings=128,
        dropout_attn=0.0,
        no_bias=False,
    )

    trainer_config = TrainerConfig(
        tensor_model_parallel_size=tensor_model_parallel_size,
        use_flash_attn=False,
    )

    init_config = InitConfig(seed=42, init_std=0.02)
    optim_config = OptimConfig(max_lr=1e-3, weight_decay=0.01)
    data_config = DataConfig()
    parallel_config = ParallelConfig()
    operation_config = OperationConfig(train_steps=100)
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
    )


def shard_tp1_to_tp2(state_dict, tp_rank, tp_size=2):
    """
    Shard TP=1 weights for TP=2.

    IMPORTANT: This implementation uses [input_size, output_size] weight shape
    (not the standard PyTorch [output_size, input_size]).

    Weight shapes in TP=1:
    - linear_q.weight: [input_size, output_size] = [512, 512]
    - linear_kv.weight: [input_size, output_size] = [512, 1024] (K and V concatenated)
    - output.weight: [input_size, output_size] = [512, 512]

    For TP=2:
    - linear_q.weight: [512, 256] (split along output dim)
    - linear_kv.weight: [512, 512] (split along output dim, K and V split separately)
    - output.weight: [256, 512] (split along input dim)

    For ColumnParallelLinear (Q, K, V): split along output dimension (dim 1)
    For RowParallelLinear (output): split along input dimension (dim 0)
    """
    sharded_state_dict = {}

    for name, param in state_dict.items():
        if 'linear_q.weight' in name:
            # Column parallel: split along output dimension (dim 1)
            # TP=1: [512, 512] -> TP=2: [512, 256]
            chunk_size = param.shape[1] // tp_size
            sharded_param = param[:, tp_rank * chunk_size:(tp_rank + 1) * chunk_size]
            sharded_state_dict[name] = sharded_param
        elif 'linear_q.bias' in name:
            # Bias splits along output dimension (dim 0)
            # TP=1: [512] -> TP=2: [256]
            chunk_size = param.shape[0] // tp_size
            sharded_param = param[tp_rank * chunk_size:(tp_rank + 1) * chunk_size]
            sharded_state_dict[name] = sharded_param
        elif 'linear_kv.weight' in name:
            # KV weight: [input_size, 2 * output_size] where K and V are concatenated along dim 1
            # TP=1: [512, 1024] -> TP=2: [512, 512]
            # K is first half, V is second half along dim 1
            # Need to split K and V separately
            k_dim = param.shape[1] // 2  # 512
            k_weight = param[:, :k_dim]  # [512, 512]
            v_weight = param[:, k_dim:]  # [512, 512]

            # Split each along output dimension (dim 1)
            chunk_size = k_weight.shape[1] // tp_size  # 256
            k_weight_sharded = k_weight[:, tp_rank * chunk_size:(tp_rank + 1) * chunk_size]  # [512, 256]
            v_weight_sharded = v_weight[:, tp_rank * chunk_size:(tp_rank + 1) * chunk_size]  # [512, 256]

            # Concatenate back
            sharded_param = torch.cat([k_weight_sharded, v_weight_sharded], dim=1)  # [512, 512]
            sharded_state_dict[name] = sharded_param
        elif 'linear_kv.bias' in name:
            # KV bias: [2 * output_size] where K and V are concatenated along dim 0
            # TP=1: [1024] -> TP=2: [512]
            k_dim = param.shape[0] // 2  # 512
            k_bias = param[:k_dim]  # [512]
            v_bias = param[k_dim:]  # [512]

            # Split each along output dimension
            chunk_size = k_bias.shape[0] // tp_size  # 256
            k_bias_sharded = k_bias[tp_rank * chunk_size:(tp_rank + 1) * chunk_size]  # [256]
            v_bias_sharded = v_bias[tp_rank * chunk_size:(tp_rank + 1) * chunk_size]  # [256]

            # Concatenate back
            sharded_param = torch.cat([k_bias_sharded, v_bias_sharded], dim=0)  # [512]
            sharded_state_dict[name] = sharded_param
        elif 'output.weight' in name:
            # Row parallel: split along input dimension (dim 0)
            # TP=1: [512, 512] -> TP=2: [256, 512]
            chunk_size = param.shape[0] // tp_size
            sharded_param = param[tp_rank * chunk_size:(tp_rank + 1) * chunk_size]
            sharded_state_dict[name] = sharded_param
        elif 'output.bias' in name:
            # Bias is NOT sharded in row parallel (replicated across ranks)
            sharded_state_dict[name] = param
        else:
            # Other parameters (like layer norm) are not sharded
            sharded_state_dict[name] = param

    return sharded_state_dict


def save_tp1_weights():
    """Create and save TP=1 attention weights."""
    print("\n" + "="*70)
    print("MODE: Save TP=1 Attention Weights")
    print("="*70)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Initialize TP=1
    parallel_states.initialize_model_parallel(tensor_model_parallel_size=1, timeout_in_minutes=10.0)
    config = create_config(tensor_model_parallel_size=1)

    # Set seed
    import random
    seed = 42
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Create attention layer
    attention = Attention(config).to(device)
    attention.init_weights()

    # Create sample input
    batch_size = 2
    seq_len = 64
    hidden_states = torch.randn(
        batch_size, seq_len, config.model.d_model,
        device=device, requires_grad=True
    )
    attention_mask = torch.tril(
        torch.ones(seq_len, seq_len, device=device)
    ).unsqueeze(0).unsqueeze(0).expand(batch_size, -1, -1, -1)

    # Forward pass
    print("Running forward pass...")
    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        output = attention(hidden_states, attention_mask)

    mem_peak = torch.cuda.max_memory_allocated() / 1024**2
    param_count = sum(p.numel() for p in attention.parameters())

    # Save checkpoint
    os.makedirs("checkpoints/tp1", exist_ok=True)
    checkpoint_path = "checkpoints/tp1/attention.pt"
    torch.save({
        'state_dict': attention.state_dict(),
        'config': config,
        'hidden_states': hidden_states,
        'attention_mask': attention_mask,
        'output': output,
    }, checkpoint_path)

    print(f"\nTP=1 Statistics:")
    print(f"  Parameters: {param_count:,}")
    print(f"  Peak memory: {mem_peak:.2f} MB")
    print(f"  Output norm: {output.norm().item():.6f}")
    print(f"  Output mean: {output.mean().item():.6f}")
    print(f"\nModel saved to {checkpoint_path}")

    return checkpoint_path, output


def load_and_compare_tp2():
    """Load TP=1 weights into TP=2 and compare."""
    print("\n" + "="*70)
    print("MODE: Load TP=1 Weights into TP=2 and Compare")
    print("="*70)

    # Get rank info
    if 'RANK' not in os.environ:
        print("ERROR: Must run with torchrun for TP=2")
        print("Usage: torchrun --nproc_per_node=2 tests/test_tp_weight_sharding.py --mode load_and_compare")
        sys.exit(1)

    rank = int(os.environ['RANK'])
    local_rank = int(os.environ['LOCAL_RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    device = torch.device(f'cuda:{local_rank}')

    # Initialize distributed
    dist.init_process_group(backend='nccl')
    torch.cuda.set_device(local_rank)

    # Load checkpoint
    checkpoint_path = "checkpoints/tp1/attention.pt"
    if not os.path.exists(checkpoint_path):
        print(f"ERROR: Checkpoint not found: {checkpoint_path}")
        print("Run first with: python tests/test_tp_weight_sharding.py --mode save_weights")
        sys.exit(1)

    if rank == 0:
        print(f"Loading checkpoint from {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Create TP=2 config
    config = create_config(tensor_model_parallel_size=2)

    # Initialize TP=2
    parallel_states.initialize_model_parallel(tensor_model_parallel_size=2, timeout_in_minutes=10.0)
    tp_rank = parallel_states.get_tensor_model_parallel_rank()

    # Set same seed
    import random
    seed = 42
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Create attention layer (this will initialize with random weights)
    attention = Attention(config).to(device)

    # Shard TP=1 weights and load into TP=2 model
    if rank == 0:
        print("Sharding and loading TP=1 weights into TP=2 model...")

    sharded_state_dict = shard_tp1_to_tp2(checkpoint['state_dict'], tp_rank, tp_size=2)
    attention.load_state_dict(sharded_state_dict)

    # Get the same input
    hidden_states = checkpoint['hidden_states'].to(device)
    attention_mask = checkpoint['attention_mask'].to(device)

    # Verify input is same across ranks
    input_sum = hidden_states.sum().item()
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
        output_tp2 = attention(hidden_states, attention_mask)

    # Measure memory
    mem_peak = torch.cuda.max_memory_allocated() / 1024**2
    local_params = sum(p.numel() for p in attention.parameters())

    # Get original TP=1 output
    output_tp1 = checkpoint['output'].to(device)

    # Compare outputs (both ranks should produce identical output after all-reduce)
    output_diff = (output_tp2 - output_tp1).abs().max().item()
    output_norm_diff = (output_tp2.norm() - output_tp1.norm()).abs().item()

    print(f"[Rank {rank}] TP=2 Statistics:")
    print(f"  Local params: {local_params:,}")
    print(f"  Peak memory: {mem_peak:.2f} MB")
    print(f"  Output norm: {output_tp2.norm().item():.6f}")

    # Gather results
    all_mem_peak = [torch.zeros_like(torch.tensor(mem_peak, device=device)) for _ in range(world_size)]
    all_output_diff = [torch.zeros_like(torch.tensor(output_diff, device=device)) for _ in range(world_size)]
    all_output_norm_diff = [torch.zeros_like(torch.tensor(output_norm_diff, device=device)) for _ in range(world_size)]

    dist.all_gather(all_mem_peak, torch.tensor(mem_peak, device=device))
    dist.all_gather(all_output_diff, torch.tensor(output_diff, device=device))
    dist.all_gather(all_output_norm_diff, torch.tensor(output_norm_diff, device=device))

    if rank == 0:
        print(f"\n{'='*70}")
        print(f"TP=1 vs TP=2 COMPARISON (SAME WEIGHTS, PROPERLY SHARDED)")
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
        save_tp1_weights()
    elif args.mode == 'load_and_compare':
        load_and_compare_tp2()


if __name__ == "__main__":
    main()
