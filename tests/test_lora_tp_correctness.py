#!/usr/bin/env python3
"""
Test LoRA TP correctness: TP=1 vs TP=2 output equivalence with LoRA.

This test validates:
1. TP=1 vs TP=2 produce identical outputs with LoRA enabled
2. Only LoRA parameters receive gradients (base model frozen)
3. Trainable parameter count is <5% of total with LoRA
4. LoRA adapters are properly replicated across TP ranks

Run with:
- TP=1: python tests/test_lora_tp_correctness.py --mode save_weights
- TP=2: torchrun --nproc_per_node=2 tests/test_lora_tp_correctness.py --mode load_and_compare
"""

import argparse
import os

import torch
import torch.distributed as dist

from ironcore.global_vars import set_global_states
from ironcore.language_model import LanguageModel
from ironcore.peft.utils import freeze_base_model
from lora_test_utils import (
    assert_gradient_correctness,
    cleanup_parallel,
    compare_tensors,
    create_lora_test_config,
    create_test_input,
    get_device,
    init_parallel,
    print_model_structure,
    print_parameter_stats,
    set_seed,
)


def save_tp1_model(enable_lora: bool = True):
    """Create and save TP=1 model with LoRA."""
    print("\n" + "=" * 70)
    print(f"MODE: Save TP=1 Model Weights (LoRA={'Enabled' if enable_lora else 'Disabled'})")
    print("=" * 70)

    device = get_device(0)

    # Create config and initialize
    config = create_lora_test_config(
        tp_size=1,
        enable_lora=enable_lora,
        untie_embed=True,
        xavier_init=True,
    )
    set_global_states(config)
    set_seed(42)
    init_parallel(tp_size=1)

    # Create model
    model = LanguageModel(config)
    model.to(device)

    # Freeze base model if LoRA enabled
    if config.peft.method != "none":
        freeze_base_model(model, config.peft.method)

    # Debug print model structure
    if not dist.is_initialized() or dist.get_rank() == 0:
        print_model_structure(model)

    # Count parameters
    trainable, total, lora = print_parameter_stats(model)

    # Verify trainable ratio is small
    if enable_lora:
        assert trainable / total < 0.05, (
            f"Trainable ratio too high: {100.0 * trainable / total:.2f}%"
        )
        print("✓ Trainable parameter ratio < 5%")

    # Create test input and run forward pass
    batch_size, seq_len = 2, 16
    input_ids = create_test_input(batch_size, seq_len, 100, device, seed=42)

    model.train()
    output = model(input_ids)
    loss = output.mean()
    loss.backward()

    # Check gradient flow
    assert_gradient_correctness(
        model, expect_lora_grads=enable_lora, expect_base_grads=not enable_lora
    )
    if enable_lora:
        print("✓ Only LoRA parameters receive gradients")
    else:
        print("✓ Base parameters receive gradients")

    # Save model
    os.makedirs("test_outputs/lora_tp_test", exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "output": output.detach().cpu(),
            "config": config,
        },
        "test_outputs/lora_tp_test/tp1_model.pt",
    )

    print("\n✓ TP=1 model saved successfully")
    print(f"Output shape: {output.shape}")
    print(f"Output mean: {output.mean().item():.6f}")

    cleanup_parallel(tp_size=1)


def load_and_compare_tp2(enable_lora: bool = True):
    """Load TP=1 weights into TP=2 model and compare outputs."""
    print("\n" + "=" * 70)
    print(f"MODE: Load TP=1 Weights into TP=2 (LoRA={'Enabled' if enable_lora else 'Disabled'})")
    print("=" * 70)

    # Initialize distributed
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl" if torch.cuda.is_available() else "gloo")

    rank = dist.get_rank()
    device = get_device(rank)

    # Create config and initialize
    config = create_lora_test_config(
        tp_size=2,
        enable_lora=enable_lora,
        untie_embed=True,
        xavier_init=True,
    )
    set_global_states(config)
    set_seed(42)
    init_parallel(tp_size=2)

    # Create model
    model = LanguageModel(config)
    model.to(device)

    # Freeze base model if LoRA enabled
    if config.peft.method != "none":
        freeze_base_model(model, config.peft.method)

    # Debug print model structure (only rank 0)
    if rank == 0:
        print_model_structure(model)

    # Count parameters
    if rank == 0:
        print_parameter_stats(model)

    # Load TP=1 checkpoint
    checkpoint = torch.load(
        "test_outputs/lora_tp_test/tp1_model.pt", map_location=device, weights_only=False
    )
    tp1_output = checkpoint["output"].cpu()

    # Zero out all parameters to ensure load_state_dict is actually working
    for p in model.parameters():
        p.data.zero_()

    # Load weights with proper sharding
    from ironcore.parallel.tensor_parallel import comm
    from ironcore.peft.lora import (
        LoRAColumnParallelLinear,
        LoRAConcatenatedColumnParallel,
        LoRARowParallelLinear,
    )

    model_state = {}
    for name, param in model.named_parameters():
        loaded_param = checkpoint["model"][name]

        # Detect if this parameter belongs to a LoRA wrapper
        wrapper = None
        parts = name.split(".")
        for i in range(len(parts)):
            parent_name = ".".join(parts[:i])
            if not parent_name:
                parent_module = model
            else:
                # Resolve parent_module
                parent_module = model
                for p in parts[:i]:
                    if p.isdigit():
                        parent_module = parent_module[int(p)]
                    else:
                        parent_module = getattr(parent_module, p)

            if isinstance(
                parent_module,
                (LoRAColumnParallelLinear, LoRAConcatenatedColumnParallel, LoRARowParallelLinear),
            ):
                wrapper = parent_module
                break

        if wrapper:
            if isinstance(wrapper, (LoRAColumnParallelLinear, LoRAConcatenatedColumnParallel)):
                # ColumnParallel shards WEIGHT and BIAS and LoRA_B along last dim
                if "lora_B" in name or name.endswith(".weight") or name.endswith(".bias"):
                    # Check for concatenated weights
                    concat = 1
                    if "base_layer" in name:
                        concat = getattr(wrapper.base_layer, "concatenated_weights", 1)

                    sharded = comm.split_to_model_parallel_workers(
                        loaded_param,
                        {
                            "column_parallel": True,
                            "row_parallel": False,
                            "concatenated_weights": concat,
                        },
                    )
                    model_state[name] = sharded.to(device)
                else:
                    # lora_A or other replicated params
                    model_state[name] = loaded_param.to(device)
            elif isinstance(wrapper, LoRARowParallelLinear):
                # RowParallel shards WEIGHT and LoRA_A along first dim
                if "lora_A" in name or (".weight" in name and "lora" not in name):
                    sharded = comm.split_to_model_parallel_workers(
                        loaded_param,
                        {"column_parallel": False, "row_parallel": True, "concatenated_weights": 1},
                    )
                    model_state[name] = sharded.to(device)
                else:
                    # lora_B and base_layer.bias are replicated
                    model_state[name] = loaded_param.to(device)

            # Continue to next parameter
            if name in model_state:
                continue

        # Base parameters - shard if needed
        module_name = ".".join(name.split(".")[:-1])
        module = dict(model.named_modules()).get(module_name)

        if module and hasattr(module, "column_parallel") and module.column_parallel:
            # ColumnParallel shards WEIGHT and BIAS along last dim
            sharded = comm.split_to_model_parallel_workers(
                loaded_param,
                {
                    "column_parallel": True,
                    "row_parallel": False,
                    "concatenated_weights": getattr(module, "concatenated_weights", 1),
                },
            )
            model_state[name] = sharded.to(device)
        elif module and hasattr(module, "row_parallel") and module.row_parallel:
            if "weight" in name:
                # RowParallel shards WEIGHT along FIRST dimension
                sharded = comm.split_to_model_parallel_workers(
                    loaded_param,
                    {"column_parallel": False, "row_parallel": True, "concatenated_weights": 1},
                )
                model_state[name] = sharded.to(device)
            else:
                # Bias is replicated in RowParallel
                model_state[name] = loaded_param.to(device)
        elif "word_embeddings.weight" in name:
            # VocabParallelEmbedding shards along first dimension
            sharded = comm.split_to_model_parallel_workers(
                loaded_param,
                {"column_parallel": False, "row_parallel": True, "concatenated_weights": 1},
            )
            model_state[name] = sharded.to(device)
        else:
            model_state[name] = loaded_param.to(device)

    model.load_state_dict(model_state, strict=True)

    if rank == 0:
        print("✓ Weights loaded successfully")

    # Create test input (same as TP=1)
    batch_size, seq_len = 2, 16
    input_ids = torch.empty((batch_size, seq_len), dtype=torch.long, device=device)
    if rank == 0:
        torch.manual_seed(42)  # Ensure same input as save_tp1
        input_ids.copy_(torch.randint(0, 100, (batch_size, seq_len), device=device))

    # Broadcast from rank 0 to all ranks
    dist.broadcast(input_ids, src=0)

    # Forward pass
    model.eval()
    with torch.no_grad():
        output_tp2 = model(input_ids)

    # Compare outputs
    output_tp2_cpu = output_tp2.cpu()

    if rank == 0:
        # Relaxed tolerance for bfloat16 + TP
        atol, rtol = 1e-1, 1e-1
        matches = compare_tensors(
            tp1_output,
            output_tp2_cpu,
            name="TP=1 vs TP=2 output",
            atol=atol,
            rtol=rtol,
        )
        assert matches, "Outputs differ between TP=1 and TP=2"
        print(f"✓ Outputs match within tolerance (atol={atol}, rtol={rtol})")

    cleanup_parallel(tp_size=2)

    if rank == 0:
        print("\n" + "=" * 70)
        print("ALL TESTS PASSED ✓")
        print("=" * 70)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode", choices=["save_weights", "load_and_compare"], required=True, help="Test mode"
    )
    args = parser.parse_args()

    if args.mode == "save_weights":
        save_tp1_model(enable_lora=True)
    elif args.mode == "load_and_compare":
        load_and_compare_tp2(enable_lora=True)


if __name__ == "__main__":
    main()
