"""
Direct comparison of TP=1 vs TP=2 at the same step.
Both start with the same seed and should produce identical results if TP is correct.
"""
import os
import sys
import torch
import torch.distributed as dist
import numpy as np
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ironcore.config import load_trainer_config
from ironcore.parallel import parallel_states
from ironcore.parallel.parallel import initialize_process
from ironcore.global_vars import set_global_states
from ironcore.language_model import LanguageModel
from ironcore.dataloader import get_data_iterator


def run_one_step():
    """Run exactly one training step and collect detailed statistics."""
    config = load_trainer_config()
    set_global_states(config)
    initialize_process(config)

    rank = dist.get_rank() if dist.is_initialized() else 0
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    device = torch.device(f'cuda:{rank}' if torch.cuda.is_available() else 'cpu')

    # Initialize model parallelism (required even for TP=1)
    timeout_minutes = config.parallel.timeout_minute if world_size > 1 and hasattr(config, 'parallel') and hasattr(config.parallel, 'timeout_minute') else 30
    parallel_states.initialize_model_parallel(
        tensor_model_parallel_size=config.trainer.tensor_model_parallel_size,
        timeout=timeout_minutes
    )

    tp_size = config.trainer.tensor_model_parallel_size
    tp_rank = parallel_states.get_tensor_model_parallel_rank()

    print(f"\n{'='*70}")
    print(f"[Rank {rank}] Running TP={tp_size} (TP rank {tp_rank})")
    print(f"[Rank {rank}] Precision: {config.model.precision}")
    print(f"{'='*70}")

    # Set seed
    import random
    seed = config.init.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    print(f"[Rank {rank}] Seed: {seed}")

    # Create model
    def dummy_loss_fn(loss, mask):
        return (loss * mask).sum() / mask.sum()

    model = LanguageModel(config, dummy_loss_fn).to(device)

    print(f"[Rank {rank}] Model created")

    # Create optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.optim.max_lr,
        weight_decay=config.optim.weight_decay,
        eps=1e-8,
        betas=(0.9, 0.999)
    )

    # Get data
    iterators = get_data_iterator(config)
    data_iter = iterators['train']
    batch = next(data_iter)
    input_ids = batch['input_ids'].to(device)
    labels = batch['labels'].to(device)

    print(f"[Rank {rank}] Data loaded: input_sum={input_ids.sum().item()}")

    # Verify data consistency across ranks
    if world_size > 1:
        input_sums = [None] * world_size
        dist.all_gather_object(input_sums, input_ids.sum().item())
        if rank == 0:
            if len(set(input_sums)) == 1:
                print(f"✓ Data is identical across all ranks")
            else:
                print(f"❌ ERROR: Data differs across ranks: {input_sums}")

    # Capture initial weight statistics
    weight_stats = {}
    for name, param in model.named_parameters():
        if 'layers.0.self_attention.linear_q.weight' in name:
            weight_stats['q_weight_mean_init'] = param.data.mean().item()
            weight_stats['q_weight_std_init'] = param.data.std().item()
            weight_stats['q_weight_norm_init'] = param.data.norm().item()

    # Forward pass
    print(f"[Rank {rank}] Running forward pass...")
    loss = model(input_ids, labels)
    print(f"[Rank {rank}] Loss: {loss.item():.8f}")

    # Backward pass
    print(f"[Rank {rank}] Running backward pass...")
    optimizer.zero_grad()
    loss.backward()

    # Capture gradient statistics
    grad_stats = {}
    total_grad_norm = 0.0
    for name, param in model.named_parameters():
        if param.grad is not None:
            if 'layers.0.self_attention.linear_q.weight' in name:
                grad_stats['q_weight_grad_mean'] = param.grad.mean().item()
                grad_stats['q_weight_grad_std'] = param.grad.std().item()
                grad_stats['q_weight_grad_norm'] = param.grad.norm().item()
            if 'layers.0.self_attention.linear_q.bias' in name:
                grad_stats['q_bias_grad_mean'] = param.grad.mean().item()
                grad_stats['q_bias_grad_norm'] = param.grad.norm().item()

            total_grad_norm += param.grad.norm().item() ** 2

    total_grad_norm = total_grad_norm ** 0.5
    grad_stats['total_grad_norm'] = total_grad_norm

    print(f"[Rank {rank}] Total gradient norm: {total_grad_norm:.6f}")

    # Optimizer step
    print(f"[Rank {rank}] Running optimizer step...")
    optimizer.step()

    # Capture final weight statistics
    for name, param in model.named_parameters():
        if 'layers.0.self_attention.linear_q.weight' in name:
            weight_stats['q_weight_mean_final'] = param.data.mean().item()
            weight_stats['q_weight_std_final'] = param.data.std().item()
            weight_stats['q_weight_norm_final'] = param.data.norm().item()
            weight_stats['q_weight_delta'] = weight_stats['q_weight_mean_final'] - weight_stats['q_weight_mean_init']

    # Collect statistics
    stats = {
        'rank': rank,
        'tp_size': tp_size,
        'tp_rank': tp_rank,
        'loss': loss.item(),
        'precision': config.model.precision,
        **weight_stats,
        **grad_stats,
    }

    # Gather stats from all ranks
    all_stats = [None] * world_size if world_size > 1 else [stats]
    if world_size > 1:
        dist.all_gather_object(all_stats, stats)

    # Print comparison
    if rank == 0:
        print(f"\n{'='*70}")
        print(f"RESULTS FOR TP={tp_size}")
        print(f"{'='*70}")

        # Save to file
        output_file = f"logs/tp{tp_size}_results.json"
        os.makedirs("logs", exist_ok=True)
        with open(output_file, 'w') as f:
            json.dump(all_stats, f, indent=2)
        print(f"Results saved to {output_file}")

        # Print key metrics
        print(f"\nKey Metrics:")
        print(f"  Loss (rank 0): {all_stats[0]['loss']:.8f}")
        if world_size > 1:
            print(f"  Loss (rank 1): {all_stats[1]['loss']:.8f}")
            loss_diff = abs(all_stats[0]['loss'] - all_stats[1]['loss'])
            print(f"  Loss difference: {loss_diff:.2e}")

            if loss_diff < 1e-7:
                print(f"  ✓ Losses are IDENTICAL (diff < 1e-7)")
            else:
                print(f"  ⚠️  Losses DIFFER by {loss_diff:.2e}")

        print(f"\nGradient Statistics:")
        print(f"  Total grad norm (rank 0): {all_stats[0]['total_grad_norm']:.6f}")
        if world_size > 1:
            print(f"  Total grad norm (rank 1): {all_stats[1]['total_grad_norm']:.6f}")

        print(f"\nWeight Update (linear_q.weight):")
        print(f"  Initial mean (rank 0): {all_stats[0]['q_weight_mean_init']:.8f}")
        print(f"  Final mean (rank 0): {all_stats[0]['q_weight_mean_final']:.8f}")
        print(f"  Delta (rank 0): {all_stats[0]['q_weight_delta']:.8e}")

        if world_size > 1:
            print(f"  Initial mean (rank 1): {all_stats[1]['q_weight_mean_init']:.8f}")
            print(f"  Final mean (rank 1): {all_stats[1]['q_weight_mean_final']:.8f}")
            print(f"  Delta (rank 1): {all_stats[1]['q_weight_delta']:.8e}")

    if world_size > 1:
        dist.barrier()
        dist.destroy_process_group()

    return stats


if __name__ == "__main__":
    try:
        run_one_step()
        sys.exit(0)
    except Exception as e:
        rank = dist.get_rank() if dist.is_initialized() else 0
        print(f"[Rank {rank}] ERROR: {e}")
        import traceback
        traceback.print_exc()
        if dist.is_initialized():
            dist.destroy_process_group()
        sys.exit(1)
