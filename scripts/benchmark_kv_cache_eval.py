#!/usr/bin/env python
# Copyright (c) 2025-2026 Jaegeun Han
#
# SPDX-License-Identifier: MIT
#
# Benchmark script to compare evaluation speed with and without KV cache.
# Usage: python scripts/benchmark_kv_cache_eval.py [--num-samples N] [--model gpt2-medium]

import argparse
import os
import time
import sys
from pathlib import Path

import torch
import torch.distributed as dist

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def main():
    parser = argparse.ArgumentParser(description="Benchmark KV cache for HellaSwag evaluation")
    parser.add_argument("--num-samples", type=int, default=100, help="Number of HellaSwag samples")
    parser.add_argument("--model", type=str, default="gpt2-medium", help="HuggingFace model name")
    parser.add_argument("--no-cache", action="store_true", help="Only run without KV cache")
    args = parser.parse_args()

    print(f"Benchmarking {args.model} on HellaSwag ({args.num_samples} samples)")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    if not torch.cuda.is_available():
        print("WARNING: CUDA not available, running on CPU (will be slow)")

    # Setup environment for single GPU
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("LOCAL_RANK", "0")
    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", "29504")

    # Import after env setup
    from ironcore import get_tokenizer, set_global_states
    from ironcore.config import MainConfig
    from ironcore.config.config_model import ModelConfig, KVCacheConfig, PositionalEmbeddingConfig
    from ironcore.config import (
        DataConfig, InitConfig, OptimConfig, ParallelConfig,
        OperationConfig, TrainerConfig, UtilsConfig, ProfilerConfig,
    )
    from ironcore.eval.tasks.hellaswag import HellaSwag
    from ironcore.language_model import LanguageModel
    from ironcore.parallel.parallel_states import initialize_model_parallel
    from ironcore.tokenizer import build_tokenizer
    from ironcore.checkpointing.hf_interop import load_from_huggingface

    # Download model from HuggingFace if needed
    from huggingface_hub import snapshot_download
    model_path = snapshot_download(args.model)
    print(f"Model downloaded to: {model_path}")

    results = {}

    for use_cache in [False, True]:
        if args.no_cache and use_cache:
            continue

        print(f"\n--- Running with KV cache: {use_cache} ---")

        # Load HuggingFace config to get model dimensions
        import json
        with open(Path(model_path) / "config.json") as f:
            hf_config = json.load(f)

        n_embd = hf_config.get("n_embd", 1024)
        n_head = hf_config.get("n_head", 16)
        n_layer = hf_config.get("n_layer", 24)
        n_positions = hf_config.get("n_positions", 1024)

        print(f"GPT-2 config: d_model={n_embd}, heads={n_head}, layers={n_layer}")

        # Create model config matching HF model
        kv_cache_config = KVCacheConfig(
            enabled=use_cache,
            max_batch_size=4,
            max_seq_length=n_positions,
        )

        pos_emb = PositionalEmbeddingConfig(type="absolute")  # GPT-2 uses absolute position embeddings

        model_config = ModelConfig(
            name="GPT",
            d_model=n_embd,
            num_attention_heads=n_head,
            num_attention_groups=n_head,  # MHA for GPT-2
            head_dim=n_embd // n_head,
            num_layers=n_layer,
            d_ffn=n_embd * 4,
            max_seq_len=n_positions,
            max_position_embeddings=n_positions,
            positional_embedding=pos_emb,
            kv_cache=kv_cache_config,
            vocab_name_or_path=args.model,
            tokenizer_type="gpt2",
        )

        config = MainConfig(
            model=model_config,
            trainer=TrainerConfig(
                tensor_model_parallel_size=1,
                use_flash_attn=False,
                use_kv_cache_in_eval=use_cache,
            ),
            init=InitConfig(seed=42),
            optim=OptimConfig(),
            data=DataConfig(),
            parallel=ParallelConfig(),
            operation=OperationConfig(train_steps=1),
            utils=UtilsConfig(),
            profiler=ProfilerConfig(),
        )

        # Initialize distributed
        if not dist.is_initialized():
            dist.init_process_group(backend="nccl" if torch.cuda.is_available() else "gloo")

        # Initialize model parallel
        initialize_model_parallel(1, timeout_in_minutes=10.0)

        # Reset global states if needed
        from ironcore.global_vars import GLOBAL_STATES
        if GLOBAL_STATES is not None:
            import ironcore.global_vars as gv
            gv.GLOBAL_STATES = None
        set_global_states(config)

        # Build tokenizer
        build_tokenizer(config)
        tokenizer = get_tokenizer()

        # Create model
        dtype = torch.float32
        model = LanguageModel(config).to(device=device, dtype=dtype)

        # Load weights using proper HF interop
        print(f"Loading weights from {model_path}...")
        load_result = load_from_huggingface(model_path, model, architecture="gpt2")
        print(f"Loaded: {len(load_result['loaded_keys'])} keys, "
              f"Missing: {len(load_result['missing_keys'])}, "
              f"Unexpected: {len(load_result['unexpected_keys'])}")

        # Debug: show some missing and unexpected keys
        if load_result['missing_keys']:
            print(f"Sample missing keys: {load_result['missing_keys'][:5]}")
        if load_result['unexpected_keys']:
            print(f"Sample unexpected keys: {load_result['unexpected_keys'][:5]}")

        model.eval()

        # Initialize KV cache if needed
        if use_cache and hasattr(model, "initialize_cache"):
            model.initialize_cache(batch_size=4, device=device)
            print("KV cache initialized")

        # Create evaluator
        evaluator = HellaSwag(
            tokenizer=tokenizer,
            batch_size=4,
            num_samples=args.num_samples,
            cache_dir=Path("./cache"),
        )

        # Time the evaluation
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start_time = time.perf_counter()

        result = evaluator.process(model)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed_time = time.perf_counter() - start_time

        results[use_cache] = {
            "accuracy": result["score"],
            "time": elapsed_time,
            "samples_per_sec": args.num_samples / elapsed_time,
        }

        print(f"Accuracy: {result['score']:.2f}%")
        print(f"Time: {elapsed_time:.2f}s")
        print(f"Throughput: {args.num_samples / elapsed_time:.2f} samples/sec")

        # Cleanup
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Print comparison
    if len(results) == 2:
        print("\n" + "=" * 60)
        print("COMPARISON")
        print("=" * 60)
        print(f"{'Metric':<20} {'No Cache':<15} {'With Cache':<15} {'Speedup':<10}")
        print("-" * 60)

        no_cache = results[False]
        with_cache = results[True]

        speedup_time = no_cache['time'] / with_cache['time'] if with_cache['time'] > 0 else 0
        speedup_throughput = with_cache['samples_per_sec'] / no_cache['samples_per_sec'] if no_cache['samples_per_sec'] > 0 else 0

        print(f"{'Time (s)':<20} {no_cache['time']:<15.2f} {with_cache['time']:<15.2f} {speedup_time:<10.2f}x")
        print(f"{'Samples/sec':<20} {no_cache['samples_per_sec']:<15.2f} {with_cache['samples_per_sec']:<15.2f} {speedup_throughput:<10.2f}x")
        print(f"{'Accuracy (%)':<20} {no_cache['accuracy']:<15.2f} {with_cache['accuracy']:<15.2f} {'-':<10}")

        # Verify accuracy
        acc_diff = abs(no_cache['accuracy'] - with_cache['accuracy'])
        if acc_diff < 1.0:
            print(f"\n✓ Accuracy matches within tolerance ({acc_diff:.2f}% diff)")
        else:
            print(f"\n✗ WARNING: Accuracy differs by {acc_diff:.2f}%")


if __name__ == "__main__":
    main()
