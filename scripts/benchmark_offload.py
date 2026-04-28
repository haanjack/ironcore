# Copyright (c) 2025-2026 Jaegeun Han
#
# SPDX-License-Identifier: Apache-2.0

"""
Benchmark VRAM usage with and without optimizer offloading (M1).

Runs each configuration as a subprocess to avoid CUDA caching allocator
polluting baselines between in-VRAM and offloaded runs.
"""

import json
import subprocess
import sys

BENCH_SCRIPT = """
import gc, json, torch
from torch import nn
from ironcore.optimizer.adamw import AdamWOptimizer

device = torch.device("cuda")
hidden, layers = HIDDEN, LAYERS

# Build model
layer_modules = []
param_count = 0
for _ in range(layers):
    qkv = nn.Linear(hidden, 3 * hidden, bias=False)
    out = nn.Linear(hidden, hidden, bias=False)
    ffn_up = nn.Linear(hidden, 4 * hidden, bias=False)
    ffn_down = nn.Linear(4 * hidden, hidden, bias=False)
    layer_modules.extend([qkv, out, ffn_up, ffn_down])
    param_count += sum(p.numel() for p in [qkv.weight, out.weight, ffn_up.weight, ffn_down.weight])
embed = nn.Embedding(50257, hidden)
param_count += embed.weight.numel()
model = nn.Sequential(embed, *layer_modules).to(device)

# Set grads
for p in model.parameters():
    p.grad = torch.randn_like(p)

gc.collect()
torch.cuda.synchronize()
baseline = torch.cuda.memory_allocated()

# Create optimizer and step
opt = AdamWOptimizer([{"params": model.parameters()}], lr=1e-3, offload_enabled=OFFLOAD)
opt.step()

gc.collect()
torch.cuda.synchronize()
after_step = torch.cuda.memory_allocated()

# Verify states
cpu_states = sum(1 for s in opt.state.values() if s["exp_avg"].device.type == "cpu")
gpu_states = sum(1 for s in opt.state.values() if s["exp_avg"].device.type != "cpu")

print(json.dumps({
    "params_m": param_count / 1e6,
    "baseline_mb": baseline / 1024 / 1024,
    "after_step_mb": after_step / 1024 / 1024,
    "state_delta_mb": (after_step - baseline) / 1024 / 1024,
    "expected_state_mb": param_count * 4 * 2 / 1024 / 1024,
    "cpu_states": cpu_states,
    "gpu_states": gpu_states,
}))
"""


def run_single(hidden, layers, offload):
    script = (
        BENCH_SCRIPT.replace("HIDDEN", str(hidden))
        .replace("LAYERS", str(layers))
        .replace("OFFLOAD", str(offload))
    )
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, check=False
    )
    # Find the JSON line in output
    for line in result.stdout.strip().split("\n"):
        line = line.strip()
        if line.startswith("{"):
            return json.loads(line)
    print(f"STDERR: {result.stderr[-500:]}", file=sys.stderr)
    return None


def main():
    if not _cuda_available():
        print("CUDA not available.")
        return

    print(f"Device: {_get_device_name()}")
    print()

    configs = [
        (768, 12, "GPT-2 Small (~124M)"),
        (1024, 24, "GPT-2 Medium (~350M)"),
    ]

    print(
        f"{'Model':<24} {'Params':>8} {'In-VRAM':>12} {'Offloaded':>12} {'Saved':>10} {'Expected':>10} {'% Saved':>8}"
    )
    print(
        f"{'':.<24} {'(M)':>8} {'state(MB)':>12} {'state(MB)':>12} {'(MB)':>10} {'(MB)':>10} {'':>8}"
    )
    print("-" * 90)

    for hidden, layers, label in configs:
        vram = run_single(hidden, layers, offload=False)
        off = run_single(hidden, layers, offload=True)

        if vram is None or off is None:
            print(f"{label:<24} ERROR")
            continue

        saved = vram["state_delta_mb"] - off["state_delta_mb"]
        pct = (saved / vram["state_delta_mb"] * 100) if vram["state_delta_mb"] > 0 else 0

        print(
            f"{label:<24} {vram['params_m']:>8.1f} "
            f"{vram['state_delta_mb']:>12.1f} {off['state_delta_mb']:>12.1f} "
            f"{saved:>10.1f} {vram['expected_state_mb']:>10.1f} {pct:>7.1f}%"
        )
        print(
            f"  {'':>8} "
            f"{'(states: ' + str(vram['gpu_states']) + ' on GPU)':>12} "
            f"{'(states: ' + str(off['cpu_states']) + ' on CPU)':>12}"
        )


def _cuda_available():
    result = subprocess.run(
        [sys.executable, "-c", "import torch; print(torch.cuda.is_available())"],
        capture_output=True,
        text=True,
        check=False,
    )
    return "True" in result.stdout


def _get_device_name():
    result = subprocess.run(
        [sys.executable, "-c", "import torch; print(torch.cuda.get_device_name(0))"],
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip()


if __name__ == "__main__":
    main()
