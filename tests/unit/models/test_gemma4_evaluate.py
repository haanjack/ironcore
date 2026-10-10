# Copyright (c) 2025-2026 Jaegeun Han
# SPDX-License-Identifier: Apache-2.0
"""Final-answer scoring and lossless native-to-vLLM adapter projection checks."""

import json

import pytest
import torch
from examples.gemma4_evaluate import convert_adapter, extract
from safetensors.torch import load_file, save_file
from tests.fixtures.gemma4 import gemma4_pair

from ironcore.peft import save_lora_adapter


@pytest.mark.parametrize(
    "text,n,expected",
    [
        ("C<turn|>", 10, "C"),
        ("The answer is (J).", 10, "J"),
        ("**Answer: D**", 10, "D"),
        ("<|channel>thought\nCould be A or B.\n<channel|>The answer is (C).<turn|>", 10, "C"),
        ("<|channel>thought\nThe answer is (B), but I need more time.", 10, None),
        ("<|channel>thought\nWe should try A.\n<channel|>", 10, None),
        ("J", 4, None),
        ("I cannot answer this question.", 10, None),
    ],
)
def test_only_final_answer_is_scored(text, n, expected):
    assert extract(text, n) == expected


def test_vllm_export_keeps_every_expert_and_shared_kv_adapter(monkeypatch, tmp_path):
    native, reference, _ = gemma4_pair(
        monkeypatch,
        "A4B",
        lora=True,
        lora_targets=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    with torch.no_grad():
        for name, param in native.named_parameters():
            if name.endswith("lora_B"):
                param.normal_(0, 0.1)
    source, target = tmp_path / "native", tmp_path / "vllm"
    save_lora_adapter(native, source)
    checkpoint = tmp_path / "base"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text(json.dumps(reference.config.to_dict()))
    convert_adapter(source, target, checkpoint)
    original = load_file(str(source / "adapter_model.safetensors"))
    exported = load_file(str(target / "adapter_model.safetensors"))
    mapping = json.loads((target / "conversion.json").read_text())
    assert mapping["tensor_counts"]["experts"] == 4 * 4 * 3 * 2
    assert mapping["tensor_counts"]["shared_kv_copies"] == 2 * 2
    assert {m["source"] for m in mapping["mapping"]} == original.keys()
    for m in mapping["mapping"]:
        torch.testing.assert_close(exported[m["target"]].T, original[m["source"]], rtol=0, atol=0)
    # Separate gate/up A matrices remain independent (no incorrect concatenation).
    for expert in range(4):
        for proj in ("gate_proj", "up_proj", "down_proj"):
            key = f"model.layers.0.experts.{expert}.lora_{proj}"
            hf = f"base_model.model.model.layers.0.moe.experts.{expert}.{proj}"
            a, b = original[key + ".lora_A"], original[key + ".lora_B"]
            x = torch.randn(3, a.size(0))
            expected = (x @ a) @ b
            actual = torch.nn.functional.linear(
                torch.nn.functional.linear(x, exported[hf + ".lora_A.weight"]),
                exported[hf + ".lora_B.weight"],
            )
            torch.testing.assert_close(actual, expected, rtol=1e-5, atol=2e-7)
    # Missing even one expert adapter must fail rather than silently change inference.
    original.pop("model.layers.0.experts.0.lora_up_proj.lora_A")
    save_file(original, str(source / "adapter_model.safetensors"))
    with pytest.raises(ValueError, match="Incomplete adapter coverage"):
        convert_adapter(source, tmp_path / "incomplete", checkpoint)


def test_prefetch_shim_moves_adapter_allocation_without_moving_base(monkeypatch):
    """CPU base storage must not send Triton adapter buffers to CPU."""
    import importlib.util
    import sys
    from pathlib import Path
    from types import ModuleType, SimpleNamespace

    modules = {}
    for name in (
        "vllm",
        "vllm.v1",
        "vllm.v1.worker",
        "vllm.v1.worker.gpu_worker",
        "vllm.lora",
        "vllm.lora.layers",
        "vllm.lora.layers.utils",
        "vllm.lora.layers.column_parallel_linear",
    ):
        module = ModuleType(name)
        module.__path__ = []
        modules[name] = module
        monkeypatch.setitem(sys.modules, name, module)
    utils = modules["vllm.lora.layers.utils"]
    alias = modules["vllm.lora.layers.column_parallel_linear"]
    modules["vllm.lora.layers"].utils = utils
    cpu = SimpleNamespace(weight=torch.ones(2, 2))
    cuda = SimpleNamespace(weight=SimpleNamespace(device=torch.device("cuda:1")))

    def original(layer):
        return layer.weight.device

    utils._get_lora_device = original
    alias._get_lora_device = original

    class FakeWorker:
        def load_model(self, marker):
            self.placements = [
                utils._get_lora_device(cpu),
                alias._get_lora_device(cpu),
                utils._get_lora_device(cuda),
            ]
            return marker

    modules["vllm.v1.worker.gpu_worker"].Worker = FakeWorker
    path = Path(__file__).parents[3] / "examples/gemma4_eval_worker.py"
    spec = importlib.util.spec_from_file_location("isolated_prefetch_shim_test", path)
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    worker = loaded.PrefetchLoRAWorker()
    worker.rank, worker.device = 1, torch.device("cuda:1")
    marker = object()
    assert worker.load_model(marker) is marker
    assert worker.placements == [torch.device("cuda:1")] * 3
    assert cpu.weight.device.type == "cpu"
    torch.testing.assert_close(cpu.weight, torch.ones(2, 2))
