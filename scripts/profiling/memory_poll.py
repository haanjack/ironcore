#!/usr/bin/env python3
# Copyright (c) 2025-2026 Jaegeun Han
#
# SPDX-License-Identifier: Apache-2.0
"""Poll GPU and host RAM every N seconds and write to a CSV.

Usage:
    python scripts/memory_poll.py --out /tmp/mem_poll.csv --interval 5
"""

import argparse
import csv
import signal
import time

import psutil

try:
    import pynvml

    pynvml.nvmlInit()
    _GPU_COUNT = pynvml.nvmlDeviceGetCount()
    _NVML_OK = True
except Exception:
    _NVML_OK = False
    _GPU_COUNT = 0


def _gpu_samples():
    """Return list of (used_mib, reserved_mib) per GPU via NVML."""
    rows = []
    if not _NVML_OK:
        return rows
    for i in range(_GPU_COUNT):
        h = pynvml.nvmlDeviceGetHandleByIndex(i)
        info = pynvml.nvmlDeviceGetMemoryInfo(h)
        rows.append((info.used // 1024**2, info.total // 1024**2))
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="/tmp/mem_poll.csv")
    parser.add_argument("--interval", type=float, default=5.0)
    args = parser.parse_args()

    gpu_headers = []
    for i in range(_GPU_COUNT):
        gpu_headers += [f"gpu{i}_used_mib", f"gpu{i}_total_mib"]

    fieldnames = ["ts", "elapsed_s", "host_used_gib", "host_total_gib"] + gpu_headers

    stop = False

    def _handle_sig(sig, frame):
        nonlocal stop
        stop = True

    signal.signal(signal.SIGINT, _handle_sig)
    signal.signal(signal.SIGTERM, _handle_sig)

    t0 = time.time()
    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        f.flush()

        print(f"[memory_poll] Writing to {args.out} every {args.interval}s", flush=True)
        while not stop:
            now = time.time()
            vm = psutil.virtual_memory()
            row = {
                "ts": time.strftime("%H:%M:%S"),
                "elapsed_s": round(now - t0, 1),
                "host_used_gib": round(vm.used / 1024**3, 2),
                "host_total_gib": round(vm.total / 1024**3, 2),
            }
            for i, (used, total) in enumerate(_gpu_samples()):
                row[f"gpu{i}_used_mib"] = used
                row[f"gpu{i}_total_mib"] = total

            writer.writerow(row)
            f.flush()

            label = f"[{row['ts']} +{row['elapsed_s']}s] host={row['host_used_gib']:.2f}GiB"
            for i in range(_GPU_COUNT):
                label += f"  GPU{i}={row[f'gpu{i}_used_mib']}MiB"
            print(label, flush=True)

            time.sleep(args.interval)

    if _NVML_OK:
        pynvml.nvmlShutdown()
    print(f"[memory_poll] Done. CSV at {args.out}", flush=True)


if __name__ == "__main__":
    main()
