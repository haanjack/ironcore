#!/usr/bin/env python3
"""PCIe/NVLink bandwidth benchmark for offload operations.

Measures effective bandwidth for:
- H2D transfers (weight streaming)
- D2H transfers (activation spill, gradient offload)
- Concurrent all-reduce + H2D (simulating DDP + offload)

Usage:
    # Basic benchmark
    python scripts/benchmark_offload_pcie.py

    # Compare NVLink on vs off
    NCCL_P2P_DISABLE=1 python scripts/benchmark_offload_pcie.py
    NCCL_P2P_DISABLE=0 python scripts/benchmark_offload_pcie.py

Output:
    JSON report with bandwidth measurements (GB/s) and transfer times.
"""

import argparse
import json
import os
import time

import torch
import torch.distributed as dist

from ironcore.offload.transfer_engine import MemoryTransferEngine
from ironcore.utils.offload_metrics import get_offload_metrics


def _get_tensor_size_mb(numel: int, dtype: torch.dtype) -> float:
    return (numel * dtype.itemsize) / (1024 * 1024)


def benchmark_h2d(
    engine: MemoryTransferEngine,
    size_mb: float,
    dtype: torch.dtype,
    device: torch.device,
    warmup: int = 3,
    iters: int = 10,
) -> dict:
    """Benchmark H2D (host-to-device) bandwidth."""
    src = torch.randn(int(size_mb * 1024 * 1024 / dtype.itemsize), dtype=dtype, pin_memory=True)
    dst = torch.empty_like(src, device=device)

    # Warmup
    for _ in range(warmup):
        handle = engine.submit_h2d(src, dst)
        engine.wait(handle)
        torch.cuda.synchronize()

    # Benchmark
    times = []
    for _ in range(iters):
        start = time.perf_counter_ns()
        handle = engine.submit_h2d(src, dst)
        engine.wait(handle)
        torch.cuda.synchronize()
        end = time.perf_counter_ns()
        times.append((end - start) / 1e6)  # ms

    return {
        "size_mb": size_mb,
        "dtype": str(dtype),
        "mean_ms": sum(times) / len(times),
        "min_ms": min(times),
        "max_ms": max(times),
        "bandwidth_gb_s": size_mb / 1024 / (sum(times) / len(times) / 1000),
    }


def benchmark_d2h(
    engine: MemoryTransferEngine,
    size_mb: float,
    dtype: torch.dtype,
    device: torch.device,
    warmup: int = 3,
    iters: int = 10,
) -> dict:
    """Benchmark D2H (device-to-host) bandwidth."""
    src = torch.randn(int(size_mb * 1024 * 1024 / dtype.itemsize), dtype=dtype, device=device)
    dst = torch.empty_like(src, pin_memory=True)

    # Warmup
    for _ in range(warmup):
        handle = engine.submit_d2h(src, dst)
        engine.wait(handle)
        torch.cuda.synchronize()

    # Benchmark
    times = []
    for _ in range(iters):
        start = time.perf_counter_ns()
        handle = engine.submit_d2h(src, dst)
        engine.wait(handle)
        torch.cuda.synchronize()
        end = time.perf_counter_ns()
        times.append((end - start) / 1e6)  # ms

    return {
        "size_mb": size_mb,
        "dtype": str(dtype),
        "mean_ms": sum(times) / len(times),
        "min_ms": min(times),
        "max_ms": max(times),
        "bandwidth_gb_s": size_mb / 1024 / (sum(times) / len(times) / 1000),
    }


def benchmark_concurrent_h2d_allreduce(
    size_mb: float,
    dtype: torch.dtype,
    device: torch.device,
    world_size: int = 1,
    warmup: int = 3,
    iters: int = 10,
) -> dict:
    """Benchmark concurrent H2D + all-reduce (simulates DDP + offload)."""

    # Setup for all-reduce
    if world_size > 1 and dist.is_initialized():
        group = dist.distributed.GroupMember.WORLD
    else:
        group = None

    src = torch.randn(int(size_mb * 1024 * 1024 / dtype.itemsize), dtype=dtype, pin_memory=True)
    dst = torch.empty_like(src, device=device)
    temp = torch.empty_like(src, device=device)

    engine = MemoryTransferEngine(device=device, enable_telemetry=False)

    # Warmup
    for _ in range(warmup):
        handle = engine.submit_h2d(src, dst)
        engine.wait(handle)
        if group is not None:
            temp.copy_(dst)
            dist.all_reduce(temp, group=group)
        torch.cuda.synchronize()

    # Benchmark
    times = []
    for _ in range(iters):
        start = time.perf_counter_ns()
        handle = engine.submit_h2d(src, dst)
        if group is not None:
            # Start all-reduce before H2D completes (simulates overlap)
            temp.copy_(dst)  # May get partial data, but we're testing bandwidth
            dist.all_reduce(temp, group=group)
        engine.wait(handle)
        torch.cuda.synchronize()
        end = time.perf_counter_ns()
        times.append((end - start) / 1e6)

    effective_bandwidth = size_mb / 1024 / (sum(times) / len(times) / 1000)

    return {
        "size_mb": size_mb,
        "dtype": str(dtype),
        "world_size": world_size,
        "mean_ms": sum(times) / len(times),
        "bandwidth_gb_s": effective_bandwidth,
    }


def main():
    parser = argparse.ArgumentParser(description="PCIe/NVLink offload benchmark")
    parser.add_argument(
        "--sizes", type=float, nargs="+", default=[10, 50, 100, 500], help="Transfer sizes in MB"
    )
    parser.add_argument("--output", type=str, help="Output JSON path")
    parser.add_argument("--device", type=str, default="cuda:0", help="CUDA device")
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["float32", "float16", "bfloat16"],
        help="Tensor dtype",
    )
    args = parser.parse_args()

    dtype_map = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}
    dtype = dtype_map[args.dtype]
    device = torch.device(args.device)

    # Check NVLink status
    nvlink_enabled = os.getenv("NCCL_P2P_DISABLE", "1") == "0"
    device_props = torch.cuda.get_device_properties(device)
    device_name = device_props.name if device_props else "unknown"

    print(f"Benchmarking offload performance on {device_name}")
    print(f"NVLink: {'enabled' if nvlink_enabled else 'disabled'}")
    print(f"Sizes: {args.sizes} MB")

    engine = MemoryTransferEngine(device=device, enable_telemetry=True)

    results = {
        "device": device_name,
        "nvlink_enabled": nvlink_enabled,
        "dtype": args.dtype,
        "benchmarks": {"h2d": [], "d2h": []},
    }

    for size_mb in args.sizes:
        print(f"\nBenchmarking {size_mb} MB...")
        h2d_result = benchmark_h2d(engine, size_mb, dtype, device)
        d2h_result = benchmark_d2h(engine, size_mb, dtype, device)
        results["benchmarks"]["h2d"].append(h2d_result)
        results["benchmarks"]["d2h"].append(d2h_result)
        print(f"  H2D: {h2d_result['bandwidth_gb_s']:.2f} GB/s")
        print(f"  D2H: {d2h_result['bandwidth_gb_s']:.2f} GB/s")

    # Concurrent benchmark (if multi-GPU)
    if dist.is_initialized() and dist.get_world_size() > 1:
        world_size = dist.get_world_size()
        print(f"\nBenchmarking concurrent H2D + all-reduce (world_size={world_size})...")
        concurrent_result = benchmark_concurrent_h2d_allreduce(
            args.sizes[0], dtype, device, world_size
        )
        results["benchmarks"]["concurrent"] = concurrent_result
        print(f"  Concurrent: {concurrent_result['bandwidth_gb_s']:.2f} GB/s")

    # Print metrics summary
    metrics = get_offload_metrics()
    print(f"\nTelemetry: {metrics}")

    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
