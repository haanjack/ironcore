# Copyright (c) 2025-2026 Jaegeun Han
#
# SPDX-License-Identifier: Apache-2.0
"""Training accuracy validation: compare loss curves across offload modes.

Runs N steps with a 130M model and real openwebtext data under each offload
configuration. All modes should converge to similar loss given the same data.

Usage:
    python scripts/training_accuracy_validation.py
    python scripts/training_accuracy_validation.py --mode baseline
    python scripts/training_accuracy_validation.py --steps 1000 --output results.json
"""

import argparse
import glob
import json
import os
import re
import shutil
import subprocess
import sys
import time

import torch
import yaml

BASE_CONFIG = "configs/validate_training_accuracy.yaml"
GEN_PREFIX = "configs/e2e_accuracy_"

MODES = {
    "baseline": {},
    "m1": {"offload.enabled": True, "offload.optimizer_offload": True},
    "m2": {"offload.enabled": True, "offload.weight_offload": True},
    "m3": {"offload.enabled": True, "offload.optimizer_offload": True, "offload.activation_spill": True, "offload.activation_spill_granularity": "sub_layer"},
    "m1m2": {"offload.enabled": True, "offload.optimizer_offload": True, "offload.weight_offload": True},
    "m1m3": {"offload.enabled": True, "offload.optimizer_offload": True, "offload.activation_spill": True, "offload.activation_spill_granularity": "sub_layer"},
    "m2m3": {"offload.enabled": True, "offload.weight_offload": True, "offload.activation_spill": True, "offload.activation_spill_granularity": "sub_layer"},
    "full": {"offload.enabled": True, "offload.optimizer_offload": True, "offload.weight_offload": True, "offload.activation_spill": True, "offload.activation_spill_granularity": "sub_layer"},
}


def _nested(flat):
    r = {}
    for k, v in flat.items():
        parts = k.split(".")
        d = r
        for p in parts[:-1]:
            d = d.setdefault(p, {})
        d[parts[-1]] = v
    return r


def _deep_merge(base, override):
    result = dict(base)
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = v
    return result


def run_mode(mode_name, overrides, steps=1000, seq_len=128, timeout=1800):
    with open(BASE_CONFIG) as f:
        base = yaml.safe_load(f)

    base["operation"]["train_steps"] = steps
    base["data"]["seq_length"] = seq_len
    base["trainer"]["log_interval"] = max(1, steps // 20)  # ~20 log lines

    nested = _nested(overrides)
    config = _deep_merge(base, nested)

    cfg_path = f"{GEN_PREFIX}{mode_name}.yaml"
    with open(cfg_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    # Clean stale checkpoints
    mp = config.get("trainer", {}).get("model_path", "")
    if mp and os.path.isdir(mp):
        for e in os.listdir(mp):
            p = os.path.join(mp, e)
            if os.path.isdir(p) and e.startswith("step_"):
                shutil.rmtree(p, ignore_errors=True)
            elif e in ("latest_step.txt", "config.json"):
                os.remove(p)

    cmd = [sys.executable, "-m", "torch.distributed.run", "--nproc_per_node=1",
           "-m", "ironcore", "train", "--config", cfg_path]
    t0 = time.perf_counter()
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, check=False)
    except subprocess.TimeoutExpired:
        return {"mode": mode_name, "status": "TIMEOUT"}
    wall = round(time.perf_counter() - t0, 1)

    combined = (proc.stdout or "") + "\n" + (proc.stderr or "")
    losses, vrams = [], []
    for line in combined.split("\n"):
        if "step:" in line and "loss:" in line:
            m = re.search(r"loss:\s*([\d.]+)", line)
            if m:
                losses.append(float(m.group(1)))
            m = re.search(r"vram:\s*([\d.]+)MB", line)
            if m:
                vrams.append(float(m.group(1)))

    if not losses:
        err = combined[-800:] if combined else "no output"
        return {"mode": mode_name, "status": "FAIL", "error": err[-500:], "wall_s": wall}

    return {
        "mode": mode_name, "status": "PASS",
        "losses": losses, "final_loss": round(losses[-1], 4),
        "min_loss": round(min(losses), 4),
        "vram_peak_mb": round(max(vrams)) if vrams else 0,
        "wall_s": wall, "steps": len(losses),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=list(MODES.keys()))
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--output", default="training_accuracy_results.json")
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument("--keep-configs", action="store_true")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("CUDA required")
        return

    modes = {args.mode: MODES[args.mode]} if args.mode else MODES
    results = []

    for name, overrides in modes.items():
        print(f"\nRunning: {name} ({args.steps} steps)...", flush=True)
        r = run_mode(name, overrides, args.steps, args.seq_len, args.timeout)
        results.append(r)
        fl = r.get("final_loss", "?")
        v = r.get("vram_peak_mb", "?")
        print(f"  [{r.get('status')}] loss={fl}  vram={v}MB  ({r.get('wall_s','?')}s)")

    baseline_loss = next((r["final_loss"] for r in results if r["mode"] == "baseline" and r.get("final_loss")), None)

    print(f"\n{'='*70}")
    print(f"TRAINING ACCURACY RESULTS ({len(results)} modes, {args.steps} steps)")
    print(f"{'='*70}")
    for r in results:
        fl = r.get("final_loss", "N/A")
        delta = ""
        if baseline_loss and r.get("final_loss") and r["mode"] != "baseline":
            d = r["final_loss"] - baseline_loss
            delta = f"  ({d:+.4f} vs baseline)"
        print(f"  {r['mode']:8s} [{r.get('status','?')}] loss={fl}{delta}  vram={r.get('vram_peak_mb','?')}MB  {r.get('wall_s','?')}s")

    with open(args.output, "w") as f:
        json.dump({"steps": args.steps, "results": results}, f, indent=2)
    print(f"\nSaved: {args.output}")

    if not args.keep_configs:
        for f in glob.glob(f"{GEN_PREFIX}*.yaml"):
            os.remove(f)


if __name__ == "__main__":
    main()
