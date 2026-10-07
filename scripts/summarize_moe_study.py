# Copyright (c) 2025-2026 Jaegeun Han
# SPDX-License-Identifier: Apache-2.0
"""Archive MoE validation/learning evidence and standalone figures."""

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
    p.add_argument("--validation", type=Path, action="append", default=[])
    p.add_argument("--ep-oracle", type=Path, required=True)
    p.add_argument("--profile", type=Path)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    study = json.loads((args.study / "study.json").read_text())
    shutil.copyfile(args.study / "study.json", args.output / "study.json")
    for directory in [*args.validation, args.ep_oracle]:
        output = args.output / directory.name
        output.mkdir(exist_ok=True)
        shutil.copyfile(directory / "report.json", output / "report.json")
        if (directory / "run.log").exists():
            shutil.copyfile(directory / "run.log", output / "run.log")
        summaries = {
            str(path.relative_to(directory)): json.loads(path.read_text())
            for path in sorted(directory.glob("*/*/rank*.json"))
        }
        (output / "rank_summaries.json").write_text(json.dumps(summaries, indent=2) + "\n")
    for job in study["jobs"]:
        path = args.study / job["label"] / "telemetry.csv"
        if path.exists():
            shutil.copyfile(path, args.output / (path.parent.name + "_telemetry.csv"))
    rows = [job for job in study["jobs"] if "ranks" in job]
    with (args.output / "measurements.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "label",
                "total_parameters",
                "active_parameters_estimate",
                "tp",
                "context",
                "micro_batch",
                "global_batch",
                "tokens_per_second",
                "peak_allocated_gib_max_rank",
                "initial_heldout_nll",
                "final_heldout_nll",
                "worst_expert_max_over_mean",
            ]
        )
        for job in rows:
            ranks = job["ranks"]
            r = ranks[0]
            writer.writerow(
                [
                    job["label"],
                    r["parameters_full"],
                    r["active_parameters_per_token_estimate"],
                    r["tp"],
                    r["context"],
                    r["micro_batch"],
                    r["global_batch"],
                    r["global_tokens_per_second"],
                    max(x["peak_allocated_bytes"] for x in ranks) / 2**30,
                    r["initial_heldout_loss"],
                    r["final_heldout_loss"],
                    max((x["max_over_mean"] for x in r["expert_stats"].values()), default=None),
                ]
            )
    if args.profile:
        for path in args.profile.glob("profile/*key_averages.csv"):
            shutil.copyfile(path, args.output / path.name)
        ranks = [json.loads(path.read_text()) for path in sorted(args.profile.glob("rank*.json"))]
        (args.output / "profile_results.json").write_text(json.dumps(ranks, indent=2) + "\n")
        profiles = []
        for path in args.profile.glob("profile/*chrome.json"):
            durations, counts = defaultdict(float), defaultdict(int)
            for event in json.loads(path.read_text())["traceEvents"]:
                if event.get("cat") == "kernel":
                    durations[event["name"]] += event.get("dur", 0)
                    counts[event["name"]] += 1
            profiles.append(
                {
                    "source": str(path),
                    "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    "kernel_count": sum(counts.values()),
                    "summed_kernel_microseconds": sum(durations.values()),
                    "kernels": [
                        {"name": name, "microseconds": duration, "count": counts[name]}
                        for name, duration in sorted(durations.items(), key=lambda x: -x[1])
                    ],
                }
            )
        (args.output / "profile_kernels.json").write_text(json.dumps(profiles, indent=2) + "\n")

    source_paths = [
        *ROOT.glob("ironcore/layers/moe/*.py"),
        *ROOT.glob("ironcore/parallel/expert_parallel/*.py"),
        ROOT / "ironcore/parallel/parallel.py",
        ROOT / "ironcore/trainers/base_trainer.py",
        ROOT / "ironcore/training_utils.py",
        *ROOT.glob("scripts/*moe*.py"),
        ROOT / "scripts/validate_trainers.py",
        ROOT / "scripts/benchmark_training.py",
        ROOT / "scripts/run_training_study.py",
        *ROOT.glob("configs/model/*moe*.yaml"),
        *ROOT.glob("configs/experiments/*moe*.yaml"),
        ROOT / "tests/unit/moe/test_training_contract.py",
    ]
    (args.output / "source_hashes.json").write_text(
        json.dumps(
            {
                str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                for path in source_paths
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

    lookup = {j["label"]: j["ranks"] for j in rows}
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    for label, name, color in [
        ("50m_moe_learn", "55.2M MoE", "#2677a6"),
        ("130m_moe_learn", "133.7M MoE", "#d26b2e"),
        ("50m_dense_control", "52.8M dense", "#62824b"),
    ]:
        ranks = lookup[label]
        r = ranks[0]
        # Subtract the mean auxiliary loss from the global reported objective.
        values = [
            x["loss"]
            - sum(peer["records"][i].get("aux_loss_local", 0) for peer in ranks) / len(ranks)
            for i, x in enumerate(r["records"])
        ]
        axes[0].plot([x["step"] for x in r["records"]], values, label=name, color=color)
        axes[0].scatter(
            [0, r["steps"]],
            [r["initial_heldout_loss"], r["final_heldout_loss"]],
            color=color,
            marker="x",
            s=40,
        )
    axes[0].set(
        xlabel="Optimizer step",
        ylabel="Language-model token NLL",
        title="TinyStories pilot (x: held-out endpoints)",
    )
    axes[0].legend(fontsize=8)
    ranks = lookup["50m_moe_learn"][0]
    stats = list(ranks["expert_stats"].items())
    matrix = [row["fractions"] for _, row in stats]
    heatmap = axes[1].imshow(matrix, vmin=0, vmax=0.5, cmap="coolwarm", aspect="auto")
    for layer, fractions in enumerate(matrix):
        for expert, fraction in enumerate(fractions):
            axes[1].text(expert, layer, f"{fraction:.1%}", ha="center", va="center", fontsize=8)
    axes[1].set(
        xlabel="Routed expert ID",
        ylabel="Layer",
        title="55.2M, DP=2: cumulative training selections",
    )
    axes[1].set_xticks(range(4))
    axes[1].set_yticks(range(len(stats)), [name.split(".")[2] for name, _ in stats])
    fig.colorbar(heatmap, ax=axes[1], label="Fraction of selections (balanced: 25%)")
    fig.savefig(output / "moe_learning_and_routing.png", dpi=160)
    fig.savefig(output / "moe_learning_and_routing.svg")
    plt.close(fig)


if __name__ == "__main__":
    main()
