# Copyright (c) 2025-2026 Jaegeun Han
#
# SPDX-License-Identifier: Apache-2.0

"""Gemma 4 cache, generation, configuration and checkpoint contracts."""

import json

import pytest
import torch
from tests.fixtures.gemma4 import gemma4_pair

from ironcore.checkpointing.hf_interop import export_to_huggingface, load_from_huggingface
from ironcore.checkpointing.native import HFConfigManager
from ironcore.checkpointing.weight_mapping import Architecture, WeightMapper, get_architecture
from ironcore.config.config_gemma4 import model_config_from_gemma4, validate_gemma4_runtime


@pytest.mark.parametrize("variant", ["E2B", "E4B", "31B"])
def test_native_trainer_updates_dense_gemma4(monkeypatch, variant):
    from tests.unit.trainers.test_training_correctness import make_trainer

    from ironcore.training_utils import forward_step, loss_func

    native, _, _ = gemma4_pair(monkeypatch, variant)
    native.loss_fn = loss_func
    native.train()
    tokens = torch.tensor([[2, 3, 4, 5, 6, 7]])
    batch = {"input_ids": tokens[:, :-1], "labels": tokens[:, 1:]}
    trainer = make_trainer(native, [batch, batch], loss_func, accumulation=2)
    trainer.forward_step_func = forward_step
    before = native.embedding.word_embeddings.weight.detach().clone()
    loss, norm, _ = trainer.train_step(0)
    assert 0 < loss < 10
    assert norm > 0
    assert not torch.equal(before, native.embedding.word_embeddings.weight)


def test_lora_starts_at_base_logits_and_native_trainer_updates_adapters(monkeypatch):
    from tests.unit.trainers.test_training_correctness import make_trainer

    from ironcore.training_utils import forward_step, loss_func

    native, reference, _ = gemma4_pair(monkeypatch, lora=True)
    tokens = torch.tensor([[2, 3, 4, 5, 6, 7]])
    native.eval()
    reference.eval()
    logits, _ = native(tokens)
    torch.testing.assert_close(logits, reference(tokens).logits, atol=2e-5, rtol=2e-5)
    base = native.embedding.word_embeddings.weight.detach().clone()
    native.train()
    native.loss_fn = loss_func
    batch = {"input_ids": tokens[:, :-1], "labels": tokens[:, 1:]}
    trainer = make_trainer(native, [batch], loss_func)
    trainer.forward_step_func = forward_step
    trainer.train_step(0)
    torch.testing.assert_close(native.embedding.word_embeddings.weight, base, rtol=0, atol=0)
    assert any(
        p.count_nonzero() for name, p in native.named_parameters() if name.endswith("lora_B")
    )


@pytest.mark.parametrize("variant", ["E2B", "E4B", "31B"])
def test_gemma4_mfu_counts_ple_and_shared_projections(monkeypatch, variant):
    from ironcore.utils.mfu import MFUCalculator

    native, _, config = gemma4_pair(monkeypatch, variant)
    calculator = MFUCalculator.from_config(config.model, config.data.vocab_size)
    assert calculator.get_num_parameters() == sum(p.numel() for p in native.parameters())
    assert calculator.compute_tflops(1, 8, 0.01) > 0


@pytest.mark.parametrize(
    "variant,hidden,layers,groups",
    [
        ("e2b", 1536, 35, 1),
        ("e4b", 2560, 42, 2),
        ("31b", 5376, 60, 16),
        ("26b-a4b", 2816, 30, 8),
    ],
)
def test_official_model_presets_parse(variant, hidden, layers, groups):
    from pathlib import Path

    import yaml

    from ironcore.config import ModelConfig

    raw = yaml.safe_load(
        (Path(__file__).parents[3] / f"configs/model/gemma4-{variant}.yaml").read_text()
    )
    model = ModelConfig()(**raw)
    model.gemma4.validate(model)
    assert model.d_model == hidden
    assert model.num_layers == layers
    assert model.num_attention_groups == groups
    assert model.head_dim == 256
    assert model.gemma4.global_head_dim == 512


@pytest.mark.parametrize("variant", ["e2b", "e4b", "31b", "26b-a4b"])
@pytest.mark.parametrize("peft_method", ["none", "lora"])
def test_official_presets_accept_tp2(variant, peft_method):
    from pathlib import Path

    import yaml
    from tests.fixtures.config_fixtures import create_test_config

    from ironcore.config import ModelConfig

    config = create_test_config()
    raw = yaml.safe_load(
        (Path(__file__).parents[3] / f"configs/model/gemma4-{variant}.yaml").read_text()
    )
    config.model = ModelConfig()(**raw)
    config.trainer.tensor_model_parallel_size = 2
    config.peft.method = peft_method
    validate_gemma4_runtime(config)


@pytest.mark.parametrize("invalid", ["ple_width", "ple_vocab", "kv_heads", "kv_projection"])
def test_gemma4_rejects_incompatible_tp_shards(invalid):
    from pathlib import Path

    import yaml
    from tests.fixtures.config_fixtures import create_test_config

    from ironcore.config import ModelConfig

    config = create_test_config()
    raw = yaml.safe_load((Path(__file__).parents[3] / "configs/model/gemma4-e2b.yaml").read_text())
    config.model = ModelConfig()(**raw)
    config.trainer.tensor_model_parallel_size = 4
    if invalid == "ple_width":
        config.model.gemma4.hidden_size_per_layer_input = 255
    elif invalid == "ple_vocab":
        config.model.gemma4.vocab_size_per_layer_input = 262143
    elif invalid == "kv_heads":
        config.model.num_attention_groups = 2
    else:
        config.model.head_dim = 258
    with pytest.raises(ValueError, match="Gemma 4"):
        validate_gemma4_runtime(config)


@pytest.mark.parametrize("variant", ["E2B", "E4B", "31B", "A4B"])
def test_cached_decode_matches_full_forward_and_generation(monkeypatch, variant):
    native, reference, _ = gemma4_pair(monkeypatch, variant)
    native.eval()
    reference.eval()
    tokens = torch.tensor([[2, 3, 4, 5, 6, 7]])
    _, cache = native(tokens[:, :4], use_cache=True)
    cached, cache = native(tokens[:, 4:], use_cache=True, past_key_values=cache)
    complete, _ = native(tokens)
    torch.testing.assert_close(cached, complete[:, 4:], atol=2e-5, rtol=2e-5)
    assert cache[0][0].size(1) == 6
    expected = reference.generate(
        tokens[:, :3], max_new_tokens=4, do_sample=False, eos_token_id=None
    )
    actual = native.generate(tokens[:, :3], max_new_tokens=4, do_sample=False)
    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("variant", ["E2B", "31B"])
@pytest.mark.parametrize("strategy", ["optimized", "standard"])
def test_activation_recomputation_preserves_gradients(monkeypatch, variant, strategy):
    native, _, config = gemma4_pair(monkeypatch, variant)
    native.train()
    tokens = torch.tensor([[2, 3, 4, 5, 6, 7]])
    actual, _ = native(tokens)
    actual.square().sum().backward()
    expected = {name: param.grad.clone() for name, param in native.named_parameters()}
    native.zero_grad(set_to_none=True)
    config.operation.activation_recompute = True
    native.model.activation_recompute = True
    native.model.use_reentrant = strategy == "optimized"
    recomputed, _ = native(tokens)
    recomputed.square().sum().backward()
    torch.testing.assert_close(recomputed, actual)
    for name, param in native.named_parameters():
        torch.testing.assert_close(
            param.grad,
            expected[name],
            atol=3e-5,
            rtol=3e-4,
            msg=lambda info, key=name: f"{key}: {info}",
        )


def test_multimodal_checkpoint_import_and_text_export(monkeypatch, tmp_path):
    native, reference, config = gemma4_pair(monkeypatch)
    source = tmp_path / "source"
    source.mkdir()
    text_config = HFConfigManager.get_hf_config(config)
    (source / "config.json").write_text(
        json.dumps({"model_type": "gemma4", "text_config": text_config})
    )
    state = {
        (
            "model.language_model." + name[len("model.") :] if name.startswith("model.") else name
        ): value
        for name, value in reference.state_dict().items()
    }
    state["model.vision_tower.unused.weight"] = torch.ones(1)
    # Official checkpoints retain these unused tensors even though newer HF
    # and native consumer layers obtain K/V from an earlier producer.
    for layer_idx in (2, 3):
        dim, heads = config.model.gemma4.head_layout(config.model, layer_idx)
        prefix = f"model.language_model.layers.{layer_idx}.self_attn"
        state[f"{prefix}.k_proj.weight"] = torch.randn(heads * dim, config.model.d_model)
        state[f"{prefix}.v_proj.weight"] = torch.randn(heads * dim, config.model.d_model)
        state[f"{prefix}.k_norm.weight"] = torch.randn(dim)
    torch.save(state, source / "pytorch_model.bin")
    loaded = load_from_huggingface(source, native, strict=True, model_config=config.model)
    assert not loaded["missing_keys"]
    exported = export_to_huggingface(
        native,
        tmp_path / "export",
        architecture="gemma4_text",
        use_safetensors=True,
        ironcore_config=config,
    )
    from transformers import Gemma4ForCausalLM

    roundtrip = Gemma4ForCausalLM.from_pretrained(exported["config_file"].parent).eval()
    native.eval()
    tokens = torch.tensor([[2, 3, 4, 5]])
    logits, _ = native(tokens)
    torch.testing.assert_close(logits, roundtrip(tokens).logits, atol=2e-5, rtol=2e-5)


@pytest.mark.parametrize("architecture", ["gemma4", "gemma4_text"])
def test_gemma4_has_a_distinct_weight_mapping(architecture):
    assert get_architecture(architecture) == Architecture.GEMMA4
    state = {"model.layers.0.self_attn.q_proj.weight": torch.arange(8).reshape(2, 4)}
    mapper = WeightMapper(Architecture.GEMMA4, 1)
    mapped = mapper.hf_to_ironcore(state)
    assert mapped[next(iter(mapped))].shape == (4, 2)
    torch.testing.assert_close(
        mapper.ironcore_to_hf(mapped)[next(iter(state))], next(iter(state.values()))
    )


@pytest.mark.parametrize("variant", ["31B", "A4B"])
def test_gemma4_config_roundtrip(monkeypatch, variant):
    _, _, config = gemma4_pair(monkeypatch, variant)
    exported = HFConfigManager.get_hf_config(config)
    recovered = model_config_from_gemma4(exported)
    assert recovered.gemma4 == config.model.gemma4
    assert recovered.head_dim == 8
    assert recovered.num_attention_groups == 2
    assert recovered.gemma4.num_global_key_value_heads == 1
    assert recovered.d_model == 16
    assert recovered.moe.use_moe == config.model.moe.use_moe


def test_a4b_packed_expert_hf_export_roundtrip(monkeypatch, tmp_path):
    native, reference, config = gemma4_pair(monkeypatch, "A4B")
    from transformers import Gemma4ForCausalLM

    exported = export_to_huggingface(
        native, tmp_path, architecture="gemma4_text", use_safetensors=True, ironcore_config=config
    )
    restored = Gemma4ForCausalLM.from_pretrained(exported["config_file"].parent).eval()
    for name, value in reference.state_dict().items():
        torch.testing.assert_close(restored.state_dict()[name], value, atol=0, rtol=0)
    tokens = torch.tensor([[2, 3, 4, 5]])
    torch.testing.assert_close(
        restored(tokens).logits, reference(tokens).logits, atol=2e-5, rtol=2e-5
    )


@pytest.mark.parametrize("unsupported", ["tp", "paged", "spill", "weight_offload"])
def test_unsupported_runtime_is_rejected_before_weight_allocation(monkeypatch, unsupported):
    _, _, config = gemma4_pair(monkeypatch)
    if unsupported == "tp":
        config.trainer.tensor_model_parallel_size = 3
    elif unsupported == "paged":
        config.model.kv_cache.use_paged = True
    elif unsupported == "spill":
        config.offload.activation_spill = True
    else:
        config.offload.weight_offload = True
    with pytest.raises(ValueError, match="Gemma 4"):
        validate_gemma4_runtime(config)


def test_moe_checkpoint_config_is_not_misidentified_as_dense(monkeypatch):
    _, reference, config = gemma4_pair(monkeypatch, "A4B")
    assert reference.config.enable_moe_block
    assert config.model.moe.use_moe
    assert config.model.moe.num_routed_experts == 4
    assert config.model.moe.expert_intermediate_size == 8


@pytest.mark.parametrize("moe", [False, True])
def test_serialized_full_attention_inherits_common_head_dim(moe):
    from transformers import Gemma4TextConfig

    reference = Gemma4TextConfig(
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        global_head_dim=8,
        num_global_key_value_heads=1,
        hidden_size_per_layer_input=0,
        enable_moe_block=moe,
        num_experts=4,
        top_k_experts=2,
        moe_intermediate_size=8,
        layer_types=["sliding_attention", "full_attention"],
    )
    serialized = reference.to_dict()
    assert "head_dim" not in serialized["per_layer_config"].get("1", {})
    native = model_config_from_gemma4(serialized)
    assert native.gemma4.global_head_dim == reference.head_dim


def test_hf_import_rejects_mismatched_kv_sharing(monkeypatch, tmp_path):
    native, reference, _ = gemma4_pair(monkeypatch)
    config = reference.config.to_dict()
    config["num_kv_shared_layers"] = 0
    (tmp_path / "config.json").write_text(json.dumps(config))
    with pytest.raises(ValueError, match="KV-sharing layout"):
        load_from_huggingface(tmp_path, native, strict=True)


def test_generate_accepts_multiple_stop_token_ids(monkeypatch):
    native, _, _ = gemma4_pair(monkeypatch)
    native.eval()
    tokens = torch.tensor([[2, 3, 4]])
    with torch.no_grad():
        logits, _ = native(tokens)
    first_token = int(logits[0, -1].argmax())
    expected = native.generate(tokens, max_new_tokens=3, eos_token_id=first_token)
    actual = native.generate(
        tokens, max_new_tokens=3, eos_token_id=[first_token, (first_token + 1) % 32]
    )
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    torch.testing.assert_close(actual, tokens, atol=0, rtol=0)


@pytest.mark.parametrize(
    "name",
    [
        "model.layers.0.self_attn.q_proj.base_layer.weight",
        "model.layers.0.experts.0.lora_gate_proj.lora_A",
        "model.layers.0.experts.0.lora_down_proj.lora_B",
    ],
)
def test_hf_export_rejects_unmerged_lora_weights(name):
    mapper = WeightMapper(Architecture.GEMMA4, 1)
    with pytest.raises(ValueError, match="Merge Gemma 4 LoRA"):
        mapper.ironcore_to_hf({name: torch.ones(2, 2)})


def test_native_checkpoint_retains_ple_weights_and_layer_scalars(monkeypatch, tmp_path):
    from ironcore.checkpointing.native import load_checkpoint, save_checkpoint

    native, _, config = gemma4_pair(monkeypatch)
    config.trainer.model_path = str(tmp_path)
    config.operation.no_save = False
    native.eval()
    with torch.no_grad():
        native.model.layers[0].layer_scalar.fill_(0.75)
    optimizer = torch.optim.AdamW(native.parameters())
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
    tokens = torch.tensor([[2, 3, 4, 5]])
    expected, _ = native(tokens)
    save_checkpoint(config, native, optimizer, scheduler, step=1)
    with torch.no_grad():
        native.model.layers[0].layer_scalar.fill_(1.0)
        native.model.embed_tokens_per_layer.weight.zero_()
    assert load_checkpoint(config, native) == 1
    actual, _ = native(tokens)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
