"""
Benchmark and Verification script for TP=1 vs TP=2.
Supports verification of correctness and profiling with Nsight Systems.
"""
import os
import sys
import torch
import torch.distributed as dist
import numpy as np
import json
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ironcore.config import load_trainer_config
from ironcore.parallel import parallel_states
from ironcore.parallel.parallel import initialize_process
from ironcore.global_vars import set_global_states
from ironcore.language_model import LanguageModel
from ironcore.dataloader import get_data_iterator
from ironcore.utils import profile_context

def run_benchmark():
    """Run benchmark loop."""
    config = load_trainer_config()
    set_global_states(config)
    initialize_process(config)

    rank = dist.get_rank() if dist.is_initialized() else 0
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    device = torch.device(f'cuda:{rank}' if torch.cuda.is_available() else 'cpu')

    # Initialize model parallelism
    timeout_minutes = config.parallel.timeout_minute if world_size > 1 and hasattr(config, 'parallel') and hasattr(config.parallel, 'timeout_minute') else 30
    parallel_states.initialize_model_parallel(
        tensor_model_parallel_size=config.trainer.tensor_model_parallel_size,
        timeout=timeout_minutes
    )

    tp_size = config.trainer.tensor_model_parallel_size
    tp_rank = parallel_states.get_tensor_model_parallel_rank()

    if rank == 0:
        print(f"\n{'='*70}")
        print(f"BENCHMARK START: TP={tp_size}")
        print(f"{'='*70}")

    # Set seed
    import random
    seed = config.init.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Create model
    def dummy_loss_fn(loss, mask):
        return (loss * mask).sum() / mask.sum()

    model = LanguageModel(config, dummy_loss_fn).to(device)

    # Create optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.optim.max_lr,
        weight_decay=config.optim.weight_decay
    )

    # Get data
    iterators = get_data_iterator(config)
    data_iter = iterators['train']

    # Warmup steps
    warmup_steps = 3
    benchmark_steps = 10

    if rank == 0:
        print(f"Running {warmup_steps} warmup steps...")

    for i in range(warmup_steps):
        batch = next(data_iter)
        input_ids = batch['input_ids'].to(device)
        labels = batch['labels'].to(device)

        loss = model(input_ids, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    torch.cuda.synchronize()
    if rank == 0:
        print(f"Running {benchmark_steps} benchmark steps...")

    # Capture initial weight statistics for a specific layer
    weight_stats = {}
    for name, param in model.named_parameters():
        if 'layers.0.self_attention.linear_q.weight' in name:
            weight_stats['q_weight_mean_init'] = param.data.mean().item()
            weight_stats['q_weight_std_init'] = param.data.std().item()
            weight_stats['q_weight_norm_init'] = param.data.norm().item()

    start_time = time.time()

    # Benchmark Loop
    losses = []
    grad_stats = {}

    for i in range(benchmark_steps):
        torch.cuda.nvtx.range_push(f"Step {i}")

        batch = next(data_iter)
        input_ids = batch['input_ids'].to(device)
        labels = batch['labels'].to(device)

        # Forward
        with profile_context("Forward"):
            loss = model(input_ids, labels)
            losses.append(loss.item())

        # Backward
        optimizer.zero_grad()
        with profile_context("Backward"):
            loss.backward()

        # Capture gradient statistics on the last step
        if i == benchmark_steps - 1:
            total_grad_norm = 0.0
            for name, param in model.named_parameters():
                if param.grad is not None:
                    if 'layers.0.self_attention.linear_q.weight' in name:
                        grad_stats['q_weight_grad_mean'] = param.grad.mean().item()
                        grad_stats['q_weight_grad_norm'] = param.grad.norm().item()
                    if 'layers.0.self_attention.linear_q.bias' in name:
                        grad_stats['q_bias_grad_mean'] = param.grad.mean().item()

                    total_grad_norm += param.grad.norm().item() ** 2
            grad_stats['total_grad_norm'] = total_grad_norm ** 0.5

        # Optimizer
        with profile_context("Optimizer"):
            optimizer.step()

        torch.cuda.nvtx.range_pop() # Step i

    torch.cuda.synchronize()
    end_time = time.time()

    # Capture final weight statistics
    for name, param in model.named_parameters():
        if 'layers.0.self_attention.linear_q.weight' in name:
            weight_stats['q_weight_mean_final'] = param.data.mean().item()
            weight_stats['q_weight_norm_final'] = param.data.norm().item()
            weight_stats['q_weight_delta'] = weight_stats['q_weight_mean_final'] - weight_stats['q_weight_mean_init']

    avg_step_time = (end_time - start_time) / benchmark_steps
    final_loss = losses[-1]

    # Collect stats
    stats = {
        'rank': rank,
        'tp_size': tp_size,
        'avg_step_time': avg_step_time,
        'final_loss': final_loss,
        'losses': losses,
        **weight_stats,
        **grad_stats,
    }

    # Gather stats
    all_stats = [None] * world_size if world_size > 1 else [stats]
    if world_size > 1:
        dist.all_gather_object(all_stats, stats)

    if rank == 0:
        print(f"\n{'='*70}")
        print(f"RESULTS TP={tp_size}")
        print(f"{'='*70}")
        print(f"Avg Step Time: {all_stats[0]['avg_step_time']*1000:.2f} ms")
        print(f"Final Loss:    {all_stats[0]['final_loss']:.8f}")

        print(f"\nGradient Statistics (Last Step):")
        print(f"  Total grad norm: {all_stats[0]['total_grad_norm']:.6f}")
        print(f"  Q-weight grad mean: {all_stats[0]['q_weight_grad_mean']:.2e}")

        print(f"\nWeight Update (linear_q.weight):")
        print(f"  Initial mean: {all_stats[0]['q_weight_mean_init']:.8f}")
        print(f"  Final mean:   {all_stats[0]['q_weight_mean_final']:.8f}")
        print(f"  Delta:        {all_stats[0]['q_weight_delta']:.2e}")

        if world_size > 1:
            loss_diff = abs(all_stats[0]['final_loss'] - all_stats[1]['final_loss'])
            print(f"\nParallel Consistency:")
            print(f"  Loss Difference (Rank 0 vs 1): {loss_diff:.2e}")
            if loss_diff < 1e-6:
                print("  ✓ CORRECTNESS CHECK PASSED")
            else:
                print("  ❌ CORRECTNESS CHECK FAILED")

        # Save results
        os.makedirs("logs", exist_ok=True)
        with open(f"logs/benchmark_tp{tp_size}.json", 'w') as f:
            json.dump(all_stats, f, indent=2)

    if world_size > 1:
        dist.barrier()
        dist.destroy_process_group()

if __name__ == "__main__":
    try:
        run_benchmark()
    except Exception as e:
        import traceback
        traceback.print_exc()
        if dist.is_initialized():
            dist.destroy_process_group()
        sys.exit(1)