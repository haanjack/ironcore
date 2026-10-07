# Copyright (c) 2025-2026 Jaegeun Han
# SPDX-License-Identifier: Apache-2.0
"""Build a standalone HTML compute/collective report from PyTorch Chrome traces.

--run-directory accepts directories with rank*.json and profile/*chrome.json.
The report keeps compact device events, source hashes and interval-union metrics;
it never treats summed kernel durations as elapsed time or network bandwidth.
"""

import argparse
import hashlib
import json
import math
import re
from collections import defaultdict
from pathlib import Path

CATEGORIES = [
    "GEMM",
    "Attention",
    "Routing / indexing",
    "Pointwise / reduction",
    "Memory / copy",
    "Collective",
    "Other kernel",
]
PHASES = ["Forward", "Backward", "Gradient norm / clip", "Optimizer", "Other / unknown"]


def merge_intervals(intervals):
    result = []
    for start, end in sorted(intervals):
        if end <= start:
            continue
        if result and start <= result[-1][1]:
            result[-1] = (result[-1][0], max(result[-1][1], end))
        else:
            result.append((start, end))
    return result


def interval_length(intervals):
    return sum(end - start for start, end in merge_intervals(intervals))


def overlap_length(left, right):
    left, right = merge_intervals(left), merge_intervals(right)
    a = b = total = 0
    while a < len(left) and b < len(right):
        total += max(0, min(left[a][1], right[b][1]) - max(left[a][0], right[b][0]))
        if left[a][1] <= right[b][1]:
            a += 1
        else:
            b += 1
    return total


def category(event):  # noqa: PLR0911 - ordered, explicit kernel classification rules
    name = event.get("name", "").lower()
    if event.get("cat") in ("gpu_memcpy", "gpu_memset"):
        return 4
    if "nccl" in name:
        return 5
    if any(x in name for x in ("flash", "fmha", "attention")):
        return 1
    if any(x in name for x in ("gemm", "cutlass", "cublas")):
        return 0
    if any(x in name for x in ("index", "scatter", "gather", "nonzero", "topk", "top_k", "scan")):
        return 2
    if any(x in name for x in ("copy", "memcpy", "memset")):
        return 4
    if any(
        x in name
        for x in (
            "elementwise",
            "reduce",
            "reduction",
            "norm",
            "softmax",
            "foreach",
            "multi_tensor",
            "adam",
        )
    ):
        return 3
    return 6


def collective_operation(name):
    for operation in ("AllReduce", "AllGather", "ReduceScatter", "Broadcast", "SendRecv", "Reduce"):
        if operation.lower() in name.lower():
            return operation
    return "Other NCCL"


def payload_bytes(event):
    sizes = {
        "c10::BFloat16": 2,
        "c10::Half": 2,
        "float": 4,
        "double": 8,
        "long int": 8,
        "int": 4,
        "unsigned char": 1,
        "bool": 1,
    }
    args = event.get("args", {})
    dims, types = args.get("Input Dims"), args.get("Input type")
    if not dims or not types or len(dims) != len(types):
        return None
    total = 0
    for shape, dtype in zip(dims, types, strict=True):
        if (
            dtype not in sizes
            or not isinstance(shape, list)
            or not all(isinstance(n, int) and n >= 0 for n in shape)
        ):
            return None
        total += math.prod(shape) * sizes[dtype]
    return total


def phase_windows(events, base_ns):
    result = []
    for event in events:
        if (
            event.get("ph") != "X"
            or "dur" not in event
            or event.get("cat") not in ("user_annotation", "python_function")
        ):
            continue
        name = event.get("name", "")
        index = None
        if name == "phase/forward":
            index = 0
        elif name == "phase/gradient_norm_clip":
            index = 2
        elif name == "phase/optimizer":
            index = 3
        elif re.search(r"torch/autograd/__init__\.py\(\d+\): backward$", name):
            index = 1
        elif name == "Forward LanguageModel":
            index = 0
        elif name.startswith("Optimizer.step#"):
            index = 3
        if index is not None:
            start = base_ns + round(event["ts"] * 1000)
            result.append(
                (start, start + round(event["dur"] * 1000), event["pid"], event["tid"], index)
            )
    return result


def launch_phase(runtime, windows, base_ns):
    if runtime is None:
        return 4
    time = base_ns + round(runtime["ts"] * 1000)
    candidates = [row for row in windows if row[0] <= time < row[1] and row[2] == runtime["pid"]]
    same_thread = [row for row in candidates if row[3] == runtime["tid"]]
    # Autograd's communication workers can launch on a different CPU thread.
    matches = same_thread or candidates
    return min(matches, key=lambda row: row[1] - row[0])[4] if matches else 4


def parse_trace(path, config):
    raw = json.loads(path.read_text())
    events = raw.pop("traceEvents")
    base = raw.get("baseTimeNanoseconds", 0)
    rank = raw.get("distributedInfo", {}).get("rank", int(re.search(r"rank(\d+)", path.name)[1]))
    windows = phase_windows(events, base)
    runtimes = {
        e["args"]["correlation"]: e
        for e in events
        if e.get("cat") in ("cuda_runtime", "cuda_driver") and "correlation" in e.get("args", {})
    }
    cpu = defaultdict(lambda: {"count": 0, "inclusive_us": 0.0})
    calls = []
    updates = []
    for event in events:
        if event.get("ph") != "X":
            continue
        name = event.get("name", "")
        if event.get("cat") == "cpu_op":
            cpu[name]["count"] += 1
            cpu[name]["inclusive_us"] += event.get("dur", 0)
        if name.startswith("nccl:") and event.get("cat") == "user_annotation":
            calls.append(
                {
                    "operation": name,
                    "cpu_us": event.get("dur", 0),
                    "input_bytes": payload_bytes(event),
                    "input_types": event.get("args", {}).get("Input type", []),
                    "input_dims": event.get("args", {}).get("Input Dims", []),
                }
            )
        if name.startswith("training_update/") and event.get("cat") == "user_annotation":
            updates.append(
                {
                    "name": name,
                    "start_ns": base + round(event["ts"] * 1000),
                    "duration_us": event.get("dur", 0),
                }
            )
    gpu = []
    for event in events:
        if event.get("cat") not in ("kernel", "gpu_memcpy", "gpu_memset") or event.get("ph") != "X":
            continue
        duration = round(event.get("dur", 0) * 1000)
        if duration <= 0:
            continue
        args = event.get("args", {})
        gpu.append(
            {
                "start_ns": base + round(event["ts"] * 1000),
                "duration_ns": duration,
                "name": event["name"],
                "category": category(event),
                "phase": launch_phase(runtimes.get(args.get("correlation")), windows, base),
                "device": args.get("device", event["pid"]),
                "stream": args.get("stream", event["tid"]),
                "kernel": event["cat"] == "kernel",
            }
        )
    if not gpu:
        raise ValueError(f"No GPU activity events: {path}")
    devices = {}
    for device in sorted({e["device"] for e in gpu}, key=str):
        selected = [e for e in gpu if e["device"] == device]
        compute = [
            (e["start_ns"], e["start_ns"] + e["duration_ns"])
            for e in selected
            if e["category"] not in (4, 5)
        ]
        comm = [
            (e["start_ns"], e["start_ns"] + e["duration_ns"])
            for e in selected
            if e["category"] == 5
        ]
        all_intervals = [(e["start_ns"], e["start_ns"] + e["duration_ns"]) for e in selected]
        begin = min(t[0] for t in all_intervals)
        end = max(t[1] for t in all_intervals)
        overlap = overlap_length(compute, comm)
        devices[str(device)] = {
            "start_ns": begin,
            "end_ns": end,
            "window_ms": (end - begin) / 1e6,
            "activity_union_ms": interval_length(all_intervals) / 1e6,
            "compute_union_ms": interval_length(compute) / 1e6,
            "collective_union_ms": interval_length(comm) / 1e6,
            "compute_collective_overlap_ms": overlap / 1e6,
            "collective_without_compute_ms": (interval_length(comm) - overlap) / 1e6,
            "no_recorded_activity_ms": ((end - begin) - interval_length(all_intervals)) / 1e6,
        }
    totals = [{"name": name, "count": 0, "sum_ms": 0.0} for name in CATEGORIES]
    kernels = defaultdict(lambda: {"count": 0, "sum_ms": 0.0, "category": 0})
    operations = defaultdict(lambda: {"count": 0, "sum_ms": 0.0})
    phases = [{"name": name, "count": 0, "sum_ms": 0.0} for name in PHASES]
    for event in gpu:
        row = totals[event["category"]]
        row["count"] += 1
        row["sum_ms"] += event["duration_ns"] / 1e6
        row = phases[event["phase"]]
        row["count"] += 1
        row["sum_ms"] += event["duration_ns"] / 1e6
        if event["kernel"]:
            row = kernels[event["name"]]
            row["count"] += 1
            row["sum_ms"] += event["duration_ns"] / 1e6
            row["category"] = event["category"]
        if event["category"] == 5:
            row = operations[collective_operation(event["name"])]
            row["count"] += 1
            row["sum_ms"] += event["duration_ns"] / 1e6
    summary = {
        "rank": rank,
        "config": config,
        "trace": str(path),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "trace_bytes": path.stat().st_size,
        "torch_metadata": {
            k: raw.get(k)
            for k in (
                "distributedInfo",
                "cuda_runtime_version",
                "cuda_driver_version",
                "record_shapes",
                "with_stack",
                "profile_memory",
                "baseTimeNanoseconds",
            )
        },
        "devices": devices,
        "categories": totals,
        "phases": phases,
        "updates": updates,
        "kernels": [
            dict(name=name, **row)
            for name, row in sorted(kernels.items(), key=lambda item: -item[1]["sum_ms"])
        ],
        "cpu_ops": [
            dict(name=name, **row)
            for name, row in sorted(cpu.items(), key=lambda item: -item[1]["inclusive_us"])
        ],
        "collective_kernels": [dict(name=name, **row) for name, row in sorted(operations.items())],
        "collective_calls": calls,
        "kernel_count": sum(e["kernel"] for e in gpu),
    }
    return summary, gpu


def build_run(directory, baseline=None):
    configs = {
        int(path.stem.removeprefix("rank")): json.loads(path.read_text())
        for path in directory.glob("rank*.json")
    }
    summaries = []
    rows = []
    for path in sorted(directory.glob("profile/*chrome.json")):
        rank = int(re.search(r"rank(\d+)", path.name)[1])
        summary, gpu = parse_trace(path, configs[rank])
        summaries.append(summary)
        rows.extend(dict(event, rank=rank) for event in gpu)
    if not summaries:
        raise ValueError(f"No exported Chrome traces: {directory}")
    origin = min(e["start_ns"] for e in rows)
    lanes = sorted({(e["rank"], str(e["device"]), str(e["stream"])) for e in rows})
    lane_index = {lane: i for i, lane in enumerate(lanes)}
    names = sorted({e["name"] for e in rows})
    name_index = {name: i for i, name in enumerate(names)}
    compact = [
        [
            round((e["start_ns"] - origin) / 1e6, 6),
            round(e["duration_ns"] / 1e6, 6),
            lane_index[(e["rank"], str(e["device"]), str(e["stream"]))],
            name_index[e["name"]],
            e["category"],
            e["phase"],
        ]
        for e in sorted(rows, key=lambda row: row["start_ns"])
    ]
    summary = {
        "label": directory.name,
        "ranks": summaries,
        "origin_ns": origin,
        "span_ms": max(e[0] + e[1] for e in compact),
        "steady_reference": json.loads((baseline / "rank0.json").read_text()) if baseline else None,
    }
    return dict(summary, lanes=lanes, names=names, events=compact)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-directory", type=Path, action="append", required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument(
        "--baseline", action="append", default=[], help="run_label=unprofiled_run_directory"
    )
    args = p.parse_args()
    baselines = {item.split("=", 1)[0]: Path(item.split("=", 1)[1]) for item in args.baseline}
    runs = [build_run(path, baselines.get(path.name)) for path in args.run_directory]
    args.output.mkdir(parents=True, exist_ok=True)
    summary = {
        "categories": CATEGORIES,
        "phases": PHASES,
        "runs": [
            {k: v for k, v in run.items() if k not in ("lanes", "names", "events")} for run in runs
        ],
    }
    (args.output / "profile_summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    def javascript_safe(value):
        if isinstance(value, dict):
            return {
                key: str(item)
                if (key.endswith("_ns") or key == "baseTimeNanoseconds") and isinstance(item, int)
                else javascript_safe(item)
                for key, item in value.items()
            }
        if isinstance(value, (list, tuple)):
            return [javascript_safe(item) for item in value]
        return value

    payload = json.dumps(
        javascript_safe({"categories": CATEGORIES, "phases": PHASES, "runs": runs}),
        ensure_ascii=False,
        separators=(",", ":"),
    ).replace("<", "\\u003c")
    template = Path(__file__).with_name("templates") / "profile_report.html"
    html = template.read_text().replace("__PROFILE_DATA__", payload)
    (args.output / "profile_report.html").write_text(html)
    print(args.output / "profile_report.html")


if __name__ == "__main__":
    main()
