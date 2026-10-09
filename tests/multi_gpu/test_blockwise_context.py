# Copyright (c) 2025-2026 Jaegeun Han
# SPDX-License-Identifier: Apache-2.0

"""Dense/MoE block execution composed with CP and optionally TP."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
from tests.multi_gpu.test_context_parallel import check_model

from ironcore.parallel import parallel_states as ps

pytestmark = [
    pytest.mark.mp,
    pytest.mark.skipif("RANK" not in os.environ, reason="torchrun required"),
]


@pytest.mark.parametrize(
    "moe,recompute,lora,expert_backend",
    [
        (False, None, False, "loop"),
        (False, "standard", True, "loop"),
        (False, "optimized", True, "loop"),
        (True, None, False, "loop"),
        (True, "standard", False, "loop"),
        (True, "optimized", False, "loop"),
        (True, None, False, "batched"),
        (True, "standard", False, "batched"),
        (True, "optimized", False, "batched"),
        (True, "optimized", True, "batched"),
        (True, None, False, "grouped"),
        (True, "standard", False, "grouped"),
        (True, "optimized", False, "grouped"),
        (True, "standard", True, "grouped"),
        (True, "optimized", True, "grouped"),
    ],
)
def test_blockwise_context(
    moe: bool, recompute: str | None, lora: bool, expert_backend: str
) -> None:
    device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    torch.cuda.set_device(device)
    if not dist.is_initialized():
        dist.init_process_group("nccl", device_id=device)
    ps.initialize_model_parallel(1, 2, context_parallel_size=2)
    check_model(
        device,
        "ring",
        lora=lora,
        moe=moe,
        mlp_chunk_size=3,
        recompute=recompute,
        compute_dtype=torch.float16,
        expert_backend=expert_backend,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    device = (
        torch.device("cuda", int(os.environ["LOCAL_RANK"]))
        if args.device == "cuda"
        else torch.device("cpu")
    )
    if device.type == "cuda":
        torch.cuda.set_device(device)
    dist.init_process_group("nccl" if device.type == "cuda" else "gloo")
    ps.initialize_model_parallel(args.tp, 2, context_parallel_size=2)
    rows = []
    for moe, recompute in [(False, None), (True, None), (True, "standard"), (True, "optimized")]:
        rows.append(
            check_model(
                device,
                "ring" if device.type == "cuda" else "sdpa",
                lora=False,
                moe=moe,
                mlp_chunk_size=3,
                recompute=recompute,
                compute_dtype=torch.float16,
            )
        )
    rows.append(
        check_model(
            device,
            "ring" if device.type == "cuda" else "sdpa",
            lora=False,
            moe=True,
            mlp_chunk_size=3,
            expert_backend="batched",
            compute_dtype=torch.float16,
        )
    )
    for recompute in ("standard", "optimized"):
        rows.append(
            check_model(
                device,
                "ring" if device.type == "cuda" else "sdpa",
                lora=True,
                mlp_chunk_size=3,
                recompute=recompute,
                compute_dtype=torch.float16,
            )
        )
    for recompute in (None, "standard", "optimized"):
        rows.append(
            check_model(
                device,
                "ring" if device.type == "cuda" else "sdpa",
                lora=False,
                moe=True,
                mlp_chunk_size=3,
                expert_backend="grouped",
                recompute=recompute,
                compute_dtype=torch.float16,
            )
        )
    for expert_backend in ("batched", "grouped"):
        rows.append(
            check_model(
                device,
                "ring" if device.type == "cuda" else "sdpa",
                lora=False,
                moe=True,
                mlp_chunk_size=3,
                expert_backend=expert_backend,
                blockwise_backend="scheduled",
                compute_dtype=torch.float16,
            )
        )
    if dist.get_rank() == 0:
        if args.report:
            args.report.write_text(json.dumps(rows, indent=2) + "\n")
        print("Dense/MoE block-wise CP trainer comparisons passed", flush=True)
    ps.destroy_model_parallel()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
