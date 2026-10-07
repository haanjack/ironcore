# Copyright (c) 2025-2026 Jaegeun Han
# SPDX-License-Identifier: Apache-2.0
"""Archive compact experiment evidence and generate blog-ready static plots."""

import argparse
import csv
import hashlib
import json
import shutil
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--study", type=Path, required=True)
    p.add_argument("--optimization", type=Path)
    p.add_argument("--validation", type=Path, action="append", default=[])
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    study = json.loads((args.study / "study.json").read_text())
    shutil.copyfile(args.study / "study.json", args.output / "study.json")
    rows = []
    for job in study["jobs"]:
        if "ranks" in job:
            rows.append((job["label"], job["ranks"]))
        path = args.study / job["label"] / "telemetry.csv"
        if path.exists():
            shutil.copyfile(path, args.output / (job["label"] + "_telemetry.csv"))
    if args.optimization:
        for path in sorted(args.optimization.iterdir()):
            results = sorted(path.glob("rank*.json"))
            if results:
                ranks = [json.loads(result.read_text()) for result in results]
                rows.append((path.name, ranks))
                (args.output / (path.name + ".json")).write_text(json.dumps(ranks, indent=2) + "\n")
            elif path.is_dir() and (path / "run.log").exists():
                shutil.copyfile(path / "run.log", args.output / (path.name + "_failure.log"))
            if (path / "telemetry.csv").exists():
                shutil.copyfile(
                    path / "telemetry.csv", args.output / (path.name + "_telemetry.csv")
                )
    for validation in args.validation:
        destination = args.output / validation.name
        destination.mkdir(exist_ok=True)
        shutil.copyfile(validation / "report.json", destination / "report.json")
        summaries = {}
        for result in sorted(validation.glob("*/*/rank*.json")):
            summaries[str(result.relative_to(validation))] = json.loads(result.read_text())
        (destination / "rank_summaries.json").write_text(json.dumps(summaries, indent=2) + "\n")
    with (args.output / "measurements.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "label",
                "model",
                "parameters_local",
                "context",
                "micro_batch",
                "global_batch",
                "loss_chunk_size",
                "compile",
                "recompute",
                "tokens_per_second",
                "mean_step_seconds",
                "peak_allocated_gib_max_rank",
                "peak_reserved_gib_max_rank",
                "initial_heldout_loss",
                "final_heldout_loss",
            ]
        )
        for label, ranks in rows:
            r = ranks[0]
            writer.writerow(
                [
                    label,
                    r["model_size_label"],
                    r["parameters_local"],
                    r["context"],
                    r["micro_batch"],
                    r["global_batch"],
                    r.get("loss_chunk_size"),
                    r["compile"],
                    r["activation_recompute"],
                    r["global_tokens_per_second"],
                    r["mean_step_seconds"],
                    max(x["peak_allocated_bytes"] for x in ranks) / 2**30,
                    max(x["peak_reserved_bytes"] for x in ranks) / 2**30,
                    r["initial_heldout_loss"],
                    r["final_heldout_loss"],
                ]
            )
    # Retain trace aggregates, not the multi-megabyte Chrome trace in the repository.
    traces = list(args.study.glob("*/profile/*chrome.json"))
    if args.optimization:
        traces += list(args.optimization.glob("*/profile/*chrome.json"))
    profiles = {}
    for path in traces:
        sums, counts = defaultdict(float), defaultdict(int)
        for event in json.loads(path.read_text())["traceEvents"]:
            if event.get("cat") == "kernel":
                sums[event["name"]] += event.get("dur", 0)
                counts[event["name"]] += 1
        profiles[path.parent.parent.name] = {
            "source": str(path),
            "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "kernel_count": sum(counts.values()),
            "summed_kernel_microseconds": sum(sums.values()),
            "kernels": [
                {"name": name, "microseconds": value, "count": counts[name]}
                for name, value in sorted(sums.items(), key=lambda item: -item[1])
            ],
        }
    (args.output / "profile_kernels.json").write_text(json.dumps(profiles, indent=2) + "\n")
    paths = [
        *ROOT.glob("ironcore/trainers/*.py"),
        ROOT / "ironcore/train.py",
        ROOT / "ironcore/training_utils.py",
        ROOT / "ironcore/alignment/dataset.py",
        ROOT / "ironcore/alignment/loss/dpo.py",
        ROOT / "ironcore/checkpointing/native.py",
        ROOT / "ironcore/language_model.py",
        ROOT / "ironcore/layers/attention.py",
        ROOT / "ironcore/config/config_trainer.py",
        ROOT / "ironcore/parallel/tensor_parallel/layers.py",
        ROOT / "ironcore/parallel/parallel.py",
        ROOT / "ironcore/parallel/grad_norm.py",
        ROOT / "ironcore/profiler.py",
        ROOT / "ironcore/utils/mfu.py",
        *ROOT.glob("scripts/*training*.py"),
        ROOT / "scripts/validate_trainers.py",
        *ROOT.glob("configs/model/cs336*.yaml"),
        *ROOT.glob("configs/experiments/cs336*.yaml"),
        ROOT / "tests/unit/trainers/test_training_correctness.py",
        ROOT / "tests/unit/test_sft_masking.py",
        ROOT / "tests/unit/profiler/test_profiler.py",
        ROOT / "tests/unit/test_mfu.py",
    ]
    (args.output / "source_hashes.json").write_text(
        json.dumps(
            {
                str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                for path in paths
            },
            indent=2,
        )
        + "\n"
    )
    plot(rows, args.output)


def plot(rows, output):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    lookup = {
        label: dict(ranks[0], peak_allocated_bytes=max(r["peak_allocated_bytes"] for r in ranks))
        for label, ranks in rows
    }
    colors = {"50m": "#2677a6", "130m": "#d26b2e"}
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    for label, name, color in [
        ("50m_learn_dp2", "52.8M, DP=2", colors["50m"]),
        ("130m_learn_dp2", "130.4M, DP=2", colors["130m"]),
        ("50m_learn_single", "52.8M, single", "#62824b"),
    ]:
        r = lookup[label]
        axes[0].plot(
            [x["step"] for x in r["records"]],
            [x["loss"] for x in r["records"]],
            label=name,
            color=color,
        )
        axes[0].scatter(
            [0, r["steps"]],
            [r["initial_heldout_loss"], r["final_heldout_loss"]],
            color=color,
            marker="x",
            s=45,
        )
    axes[0].set(
        xlabel="Optimizer step",
        ylabel="Token NLL",
        title="TinyStories pilot (x: held-out endpoints)",
    )
    axes[0].legend(fontsize=8)
    for model, color in colors.items():
        candidates = [
            (r["context"], r["global_tokens_per_second"] / 1000)
            for label, r in lookup.items()
            if label.startswith(model + "_ctx")
            and not any(x in label for x in ("compile", "recompute", "profile", "flash"))
            and r["micro_batch"] * r["context"] == 8192
        ]
        candidates.sort()
        axes[1].plot(
            [x[0] for x in candidates],
            [x[1] for x in candidates],
            "o-",
            label=model,
            color=color,
        )
    axes[1].set(
        xlabel="Context tokens",
        ylabel="Global thousands of tokens/s",
        title="Fixed 8192 microbatch tokens per GPU",
    )
    axes[1].set_xscale("log", base=2)
    axes[1].legend()
    fig.savefig(output / "learning_and_context.png", dpi=160)
    fig.savefig(output / "learning_and_context.svg")
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    for model, color in colors.items():
        candidates = [
            (
                r["micro_batch"],
                r["global_tokens_per_second"] / 1000,
                r["peak_allocated_bytes"] / 2**30,
            )
            for label, r in lookup.items()
            if label.startswith(model + "_ctx1024_mb")
        ]
        candidates.sort()
        axes[0].plot(
            [x[0] for x in candidates],
            [x[1] for x in candidates],
            "o-",
            color=color,
            label=model + ": full CE",
        )
        axes[1].plot(
            [x[0] for x in candidates],
            [x[2] for x in candidates],
            "o-",
            color=color,
            label=model + ": full CE",
        )
        optimized = [
            (
                r["micro_batch"],
                r["global_tokens_per_second"] / 1000,
                r["peak_allocated_bytes"] / 2**30,
            )
            for label, r in lookup.items()
            if label.startswith(model + "_mb") and r.get("loss_chunk_size")
        ]
        optimized.sort()
        if optimized:
            axes[0].plot(
                [x[0] for x in optimized],
                [x[1] for x in optimized],
                "s--",
                color=color,
                label=model + ": chunk CE",
            )
            axes[1].plot(
                [x[0] for x in optimized],
                [x[2] for x in optimized],
                "s--",
                color=color,
                label=model + ": chunk CE",
            )
    axes[0].set(
        xlabel="Microbatch sequences per GPU",
        ylabel="Global thousands of tokens/s",
        title="Context 1024, global batch 64",
    )
    axes[1].set(
        xlabel="Microbatch sequences per GPU",
        ylabel="Peak allocated GiB",
        title="Full CE at microbatch 32: OOM",
    )
    for axis in axes:
        axis.set_xticks([4, 8, 16, 32])
        axis.legend(fontsize=8)
        axis.grid(alpha=0.2)
    fig.savefig(output / "batch_and_memory.png", dpi=160)
    fig.savefig(output / "batch_and_memory.svg")
    plt.close(fig)


if __name__ == "__main__":
    main()
