# Copyright (c) 2025-2026 Jaegeun Han
# SPDX-License-Identifier: Apache-2.0
"""Manual interval/payload oracles for the scientific profiler report."""

import json

import pytest
from scripts.build_profile_report import (
    category,
    interval_length,
    merge_intervals,
    overlap_length,
    parse_trace,
    payload_bytes,
)


def test_interval_union_and_overlap_do_not_double_count_streams():
    compute = [(0, 10), (5, 15)]
    collective = [(8, 12), (20, 25)]
    assert merge_intervals(compute) == [(0, 15)]
    assert interval_length(compute) == 15
    assert interval_length(collective) == 9
    assert overlap_length(compute, collective) == 4
    assert interval_length(compute + collective) == 20
    assert overlap_length([(0, 10)], [(10, 20)]) == 0
    assert interval_length([(1, 1), (3, 2)]) == 0


def test_nccl_logical_tensor_bytes_are_not_wire_bytes():
    assert (
        payload_bytes(
            {"args": {"Input Dims": [[100], [1]], "Input type": ["c10::BFloat16", "long int"]}}
        )
        == 208
    )
    assert payload_bytes({"args": {"Input Dims": [[]], "Input type": ["float"]}}) == 4
    assert payload_bytes({"args": {"Input Dims": [[100]], "Input type": ["unknown"]}}) is None


@pytest.mark.parametrize(
    "name,expected",
    [
        ("ncclDevKernel_AllReduce", 5),
        ("ampere_bf16_gemm", 0),
        ("flash_bwd", 1),
        ("pytorch_flash::flash_fwd_kernel<cutlass::bfloat16_t>", 1),
        ("index_kernel", 2),
        ("copy_kernel", 4),
    ],
)
def test_kernel_classification_has_explicit_rules(name, expected):
    assert category({"name": name, "cat": "kernel"}) == expected


def test_gpu_union_and_launch_phase_against_hand_calculated_trace(tmp_path):
    def gpu(name, start, duration, correlation, stream):
        return {
            "ph": "X",
            "cat": "kernel",
            "name": name,
            "pid": 0,
            "tid": stream,
            "ts": start,
            "dur": duration,
            "args": {"device": 0, "stream": stream, "correlation": correlation},
        }

    events = [
        {
            "ph": "X",
            "cat": "user_annotation",
            "name": "phase/forward",
            "pid": 1,
            "tid": 1,
            "ts": 0,
            "dur": 100,
        },
        {
            "ph": "X",
            "cat": "cuda_runtime",
            "name": "cudaLaunchKernel",
            "pid": 1,
            "tid": 1,
            "ts": 10,
            "dur": 1,
            "args": {"correlation": 7},
        },
        {
            "ph": "X",
            "cat": "cuda_runtime",
            "name": "cudaLaunchKernel",
            "pid": 1,
            "tid": 2,
            "ts": 20,
            "dur": 1,
            "args": {"correlation": 8},
        },
        gpu("test_gemm", 30, 10, 7, 7),
        gpu("nccl_AllReduce", 35, 10, 8, 20),
        {
            "ph": "X",
            "cat": "gpu_memcpy",
            "name": "Memcpy DtoH",
            "pid": 0,
            "tid": 7,
            "ts": 50,
            "dur": 5,
            "args": {"device": 0, "stream": 7},
        },
    ]
    path = tmp_path / "profile_v0_rank0_chrome.json"
    path.write_text(json.dumps({"traceEvents": events, "baseTimeNanoseconds": 1000000}))
    summary, gpu_events = parse_trace(path, {})
    metrics = summary["devices"]["0"]
    assert metrics["window_ms"] == pytest.approx(0.025)
    assert metrics["activity_union_ms"] == pytest.approx(0.020)
    assert metrics["compute_union_ms"] == pytest.approx(0.010)
    assert metrics["collective_union_ms"] == pytest.approx(0.010)
    assert metrics["compute_collective_overlap_ms"] == pytest.approx(0.005)
    assert metrics["collective_without_compute_ms"] == pytest.approx(0.005)
    assert metrics["no_recorded_activity_ms"] == pytest.approx(0.005)
    assert [event["phase"] for event in gpu_events] == [0, 0, 4]
    assert gpu_events[0]["start_ns"] == 1030000


def test_gpu_annotations_are_not_duplicate_cpu_updates_or_collective_calls(tmp_path):
    event = {
        "ph": "X",
        "cat": "user_annotation",
        "name": "nccl:all_reduce",
        "ts": 0,
        "dur": 10,
        "pid": 1,
        "tid": 1,
        "args": {"Input Dims": [[10]], "Input type": ["c10::BFloat16"]},
    }
    update = {
        "ph": "X",
        "cat": "user_annotation",
        "name": "training_update/14",
        "ts": 0,
        "dur": 100,
        "pid": 1,
        "tid": 1,
    }
    gpu = {
        "ph": "X",
        "cat": "kernel",
        "name": "nccl_AllReduce",
        "ts": 5,
        "dur": 5,
        "pid": 0,
        "tid": 7,
        "args": {"device": 0, "stream": 7},
    }
    events = [
        event,
        dict(event, cat="gpu_user_annotation"),
        update,
        dict(update, cat="gpu_user_annotation"),
        gpu,
    ]
    path = tmp_path / "profile_v0_rank0_chrome.json"
    path.write_text(json.dumps({"traceEvents": events}))
    summary, _ = parse_trace(path, {})
    assert len(summary["updates"]) == 1
    assert len(summary["collective_calls"]) == 1
    assert summary["collective_calls"][0]["input_bytes"] == 20
