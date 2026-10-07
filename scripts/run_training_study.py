# Copyright (c) 2025-2026 Jaegeun Han
# SPDX-License-Identifier: Apache-2.0
"""Sequential real-text learning and 3090 throughput experiments.

Run after correctness validation. Every job runs alone on GPU 0/1; an OOM is
recorded as a capacity limit, while other failures stop the study. Each result
records both ranks' memory and slowest-rank synchronized step durations.

python scripts/run_training_study.py --data-dir /tmp/ironcore-corpus --output /tmp/ironcore-study
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--learning-steps", type=int, default=100)
    p.add_argument("--profile-steps", type=int, default=20)
    p.add_argument("--timeout", type=int, default=900)
    p.add_argument("--wait-for-validation", type=Path)
    p.add_argument(
        "--moe", action="store_true", help="Run the compact EP=1 MoE learning/DP/TP study"
    )
    args = p.parse_args()
    if args.output.exists():
        p.error("Use a fresh output directory")
    if args.wait_for_validation:
        import time

        deadline = time.monotonic() + args.timeout
        while True:
            if time.monotonic() > deadline:
                raise TimeoutError("Correctness validation has not completed")
            status = json.loads(args.wait_for_validation.read_text())["status"]
            if status == "passed":
                break
            if status == "failed":
                raise RuntimeError("Correctness validation failed; fix it before benchmarking")
            time.sleep(5)
    args.output.mkdir(parents=True)
    report = {"status": "running", "jobs": []}
    env = os.environ.copy()
    env.setdefault("CUDA_VISIBLE_DEVICES", "0,1")
    env["OMP_NUM_THREADS"] = "4"
    env.setdefault("TORCHINDUCTOR_CACHE_DIR", "/tmp/ironcore-inductor")
    env.setdefault("TRITON_CACHE_DIR", "/tmp/ironcore-triton")
    report_path = args.output / "study.json"

    def run_job(label, model, context, micro, batch, processes=2, steps=None, options=()):
        output = args.output / label
        output.mkdir()
        cmd = [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            f"--nproc_per_node={processes}",
            str(ROOT / "scripts/benchmark_training.py"),
            "--data-dir",
            str(args.data_dir.resolve()),
            "--output",
            str(output),
            "--model-size",
            model,
            "--context",
            str(context),
            "--micro-batch",
            str(micro),
            "--global-batch",
            str(batch),
            "--steps",
            str(steps or args.profile_steps),
            *options,
        ]
        print(f"Running {label}", flush=True)
        with (output / "telemetry.csv").open("w") as telemetry:
            monitor = subprocess.Popen(
                [
                    "nvidia-smi",
                    "--query-gpu=timestamp,index,name,temperature.gpu,power.draw,power.limit,clocks.sm,clocks.mem,utilization.gpu,memory.used",
                    "--format=csv,noheader,nounits",
                    "--loop-ms=1000",
                ],
                stdout=telemetry,
                stderr=subprocess.DEVNULL,
            )
            try:
                with (output / "run.log").open("w") as log:
                    result = subprocess.run(
                        cmd,
                        cwd=ROOT,
                        env=env,
                        stdout=log,
                        stderr=log,
                        timeout=args.timeout,
                        check=False,
                    )
            finally:
                monitor.terminate()
                monitor.wait(timeout=10)
        entry = {"label": label, "command": cmd, "returncode": result.returncode}
        if result.returncode:
            log_text = (output / "run.log").read_text()
            if "out of memory" in log_text.lower():
                entry["status"] = "oom"
            else:
                entry["status"] = "failed"
                report["jobs"].append(entry)
                report["status"] = "failed"
                report_path.write_text(json.dumps(report, indent=2) + "\n")
                raise RuntimeError(f"{label} failed: {output / 'run.log'}")
        else:
            entry["status"] = "completed"
            entry["ranks"] = [
                json.loads((output / f"rank{i}.json").read_text()) for i in range(processes)
            ]
        report["jobs"].append(entry)
        report_path.write_text(json.dumps(report, indent=2) + "\n")
        return entry

    if args.moe:
        for model in ("50m", "130m"):
            run_job(
                f"{model}_moe_learn",
                model,
                1024,
                4,
                32,
                steps=args.learning_steps,
                options=("--moe",),
            )
        run_job("50m_dense_control", "50m", 1024, 4, 32, steps=args.learning_steps)
        run_job("50m_moe_mb8", "50m", 1024, 8, 32, options=("--moe",))
        run_job("50m_moe_ctx4096", "50m", 4096, 2, 4, options=("--moe",))
        run_job("50m_moe_tp2", "50m", 1024, 4, 32, options=("--moe", "--tp", "2"))
        report["status"] = "completed"
        report_path.write_text(json.dumps(report, indent=2) + "\n")
        print(f"MoE study completed: {report_path}", flush=True)
        return

    # Identical token/global-batch budgets; separates learning from throughput.
    for model in ("50m", "130m"):
        run_job(f"{model}_learn_dp2", model, 1024, 4, 32, steps=args.learning_steps)
    run_job("50m_learn_single", "50m", 1024, 4, 32, processes=1, steps=args.learning_steps)

    # Batch-size sweep at fixed context, keeping global batch fixed within a model.
    for model in ("50m", "130m"):
        for micro in (4, 8, 16, 32):
            run_job(f"{model}_ctx1024_mb{micro}", model, 1024, micro, 64)
        # Equal microbatch token counts make context-length effects easier to interpret.
        for context, micro in ((512, 16), (2048, 4), (4096, 2), (8192, 1)):
            run_job(f"{model}_ctx{context}_mb{micro}", model, context, micro, 2 * micro)
    run_job("130m_ctx4096_recompute", "130m", 4096, 2, 4, options=("--recompute",))
    run_job("130m_ctx1024_compile", "130m", 1024, 8, 16, options=("--compile",))
    run_job("130m_ctx1024_profile", "130m", 1024, 8, 16, options=("--profile",))
    # The extension must actually be available; no silently substituted backend.
    try:
        from flash_attn import flash_attn_varlen_func
    except ImportError as error:
        report["jobs"].append(
            {"label": "130m_ctx1024_flash", "status": "unavailable", "reason": str(error)}
        )
    else:
        assert flash_attn_varlen_func is not None
        run_job("130m_ctx1024_flash", "130m", 1024, 8, 16, options=("--attention", "flash"))
    report["status"] = "completed"
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Study completed: {report_path}", flush=True)


if __name__ == "__main__":
    main()
