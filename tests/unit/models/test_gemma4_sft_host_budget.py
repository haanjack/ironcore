# Copyright (c) 2025-2026 Jaegeun Han
# SPDX-License-Identifier: Apache-2.0
"""Reject the observed unsafe full-model CP2 workload before allocation."""

import pytest
from examples.gemma4_sft import available_host_bytes, check_host_budget, estimate_host_budget

from ironcore.config.config_gemma4 import model_config_from_gemma4


def a4b_metadata():
    return model_config_from_gemma4(
        {
            "model_type": "gemma4_text",
            "vocab_size": 262144,
            "hidden_size": 2816,
            "max_position_embeddings": 262144,
            "intermediate_size": 2112,
            "num_hidden_layers": 30,
            "num_attention_heads": 16,
            "num_key_value_heads": 8,
            "head_dim": 256,
            "global_head_dim": 512,
            "num_global_key_value_heads": 2,
            "attention_k_eq_v": True,
            "hidden_size_per_layer_input": 0,
            "num_kv_shared_layers": 0,
            "enable_moe_block": True,
            "num_experts": 128,
            "top_k_experts": 8,
            "moe_intermediate_size": 704,
            "layer_types": [
                "full_attention" if (i + 1) % 6 == 0 else "sliding_attention" for i in range(30)
            ],
        }
    )


def test_real_a4b_tp2_budget_and_adapter_count():
    budget = estimate_host_budget(a4b_metadata(), 32768, 2, 2)
    assert budget["adapter_parameters"] == 333696000
    check_host_budget(budget, 112 * 1024**3)


def test_real_a4b_cp2_rejected_on_123_gib_host_before_allocation():
    model = a4b_metadata()
    tp = estimate_host_budget(model, 32768, 2, 2)
    cp = estimate_host_budget(model, 32768, 1, 2)
    assert cp["frozen_weights_bytes"] == 2 * tp["frozen_weights_bytes"]
    assert cp["spilled_inputs_bytes"] == tp["spilled_inputs_bytes"] // 2
    assert cp["required_available_bytes"] > 123 * 1024**3
    with pytest.raises(RuntimeError, match="Model allocation has not started"):
        check_host_budget(cp, 112 * 1024**3)


def test_attention_only_budget_has_previous_adapter_size():
    budget = estimate_host_budget(a4b_metadata(), 32768, 2, 2, attention_only=True)
    assert budget["adapter_parameters"] == 5744640


@pytest.mark.parametrize("cached_gib", [0, 2])
def test_container_capacity_overrides_large_physical_host_memory(monkeypatch, cached_gib):
    from pathlib import Path
    from types import SimpleNamespace

    import psutil

    monkeypatch.setattr(psutil, "virtual_memory", lambda: SimpleNamespace(available=112 * 1024**3))
    values = {
        "memory.max": str(80 * 1024**3),
        "memory.current": str(10 * 1024**3),
        "memory.stat": f"inactive_file {cached_gib * 1024**3}",
    }
    monkeypatch.setattr(Path, "read_text", lambda path: values[path.name])
    remaining = available_host_bytes()
    assert remaining == (70 + cached_gib) * 1024**3
    with pytest.raises(RuntimeError, match="Insufficient host RAM"):
        check_host_budget(estimate_host_budget(a4b_metadata(), 32768, 2, 2), remaining)


def test_active_checkpoint_cache_is_reclaimable_but_shared_offload_storage_is_not(monkeypatch):
    from pathlib import Path
    from types import SimpleNamespace

    import psutil

    monkeypatch.setattr(psutil, "virtual_memory", lambda: SimpleNamespace(available=112 * 1024**3))
    values = {
        "memory.max": str(96 * 1024**3),
        "memory.current": str(60 * 1024**3),
        "memory.stat": f"inactive_file {12 * 1024**3}\nactive_file {8 * 1024**3}\n"
        f"shmem {30 * 1024**3}\nactive_anon {40 * 1024**3}",
    }
    monkeypatch.setattr(Path, "read_text", lambda path: values[path.name])
    assert available_host_bytes() == 56 * 1024**3
