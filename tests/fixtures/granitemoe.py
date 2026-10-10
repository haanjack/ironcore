# Copyright (c) 2025-2026 Jaegeun Han
# SPDX-License-Identifier: Apache-2.0
"""Stock HF + PEFT oracle with views into packed expert parameter leaves."""

from dataclasses import dataclass

import torch


@dataclass
class AdapterBinding:
    native: torch.nn.Parameter
    reference: torch.nn.Parameter
    expert: int | None = None
    factor: str = "A"
    experts: int = 1

    def view(self, value):
        if self.expert is None:
            return value.T
        if self.factor == "A":
            return value.reshape(self.experts, -1, value.shape[-1])[self.expert].T
        return value.reshape(value.shape[0], -1, self.experts)[:, :, self.expert].T

    def reference_value(self, gradient=False):
        value = self.reference.grad if gradient else self.reference
        return None if value is None else self.view(value)


def adapter_bindings(native, reference):
    pairs = {}
    for layer, (n, h) in enumerate(
        zip(native.model.layers, reference.base_model.model.model.layers, strict=True)
    ):
        adapters = {"q_proj": n.linear_q.lora, "o_proj": n.attn_output.lora}
        for index, name in enumerate(("k_proj", "v_proj")):
            adapters[name] = n.linear_kv.lora_adapters[n.linear_kv.adapter_map[index]]
        for name, adapter in adapters.items():
            other = getattr(h.self_attn, name)
            for factor in ("A", "B"):
                pairs[f"{layer}.{name}.{factor}"] = AdapterBinding(
                    getattr(adapter, f"lora_{factor}"),
                    getattr(other, f"lora_{factor}")["default"].weight,
                )
        wrappers = {
            module.parameter_name: module
            for module in h.block_sparse_moe.experts.modules()
            if getattr(module, "parameter_name", None) in {"gate_up_proj", "down_proj"}
        }
        for expert, ne in enumerate(n.mlp.routed_experts):
            for name, wrapper in wrappers.items():
                for factor in ("A", "B"):
                    pairs[f"{layer}.expert{expert}.{name}.{factor}"] = AdapterBinding(
                        getattr(getattr(ne, f"lora_{name}"), f"lora_{factor}"),
                        getattr(wrapper, f"lora_{factor}")["default"].weight,
                        expert=expert,
                        factor=factor,
                        experts=len(n.mlp.routed_experts),
                    )
    assert {id(p.native) for p in pairs.values()} == {
        id(p) for p in native.parameters() if p.requires_grad
    }
    assert {id(p.reference) for p in pairs.values()} == {
        id(p) for p in reference.parameters() if p.requires_grad
    }
    assert sum(p.native.numel() for p in pairs.values()) == sum(
        p.numel() for p in reference.parameters() if p.requires_grad
    )
    return pairs


def granite_pair(
    monkeypatch,
    backend="grouped",
    lora=False,
    experts_implementation="eager",
    tied=True,
    hidden_size=32,
    intermediate_size=16,
    lora_rank=2,
):
    import logging
    from types import SimpleNamespace

    from tests.fixtures.config_fixtures import create_test_config
    from transformers import GraniteMoeConfig, GraniteMoeForCausalLM

    from ironcore import global_vars
    from ironcore.checkpointing.weight_mapping import Architecture, WeightMapper
    from ironcore.config.config_granitemoe import model_config_from_granitemoe
    from ironcore.language_model import LanguageModel
    from ironcore.parallel import parallel_states
    from ironcore.peft.utils import freeze_base_model
    from ironcore.utils import Timer

    hf = GraniteMoeConfig(
        vocab_size=32,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        num_local_experts=4,
        num_experts_per_tok=2,
        max_position_embeddings=32,
        tie_word_embeddings=tied,
        embedding_multiplier=12.0,
        attention_multiplier=0.125,
        residual_multiplier=0.22,
        logits_scaling=6.0,
    )
    hf._attn_implementation = "sdpa"
    hf._experts_implementation = experts_implementation
    reference = GraniteMoeForCausalLM(hf).float()
    config = create_test_config(precision="float32", use_flash_attn=False)
    config.model = model_config_from_granitemoe(hf.to_dict())
    config.model.precision = "float32"
    config.model.moe.expert_backend = backend
    config.model.moe.grouped_token_budget = 7
    config.data.vocab_size = 32
    config.data.task_type = "sft"
    config.operation.activation_recompute = False
    if lora:
        config.peft.method = "lora"
        config.peft.lora.r = lora_rank
        config.peft.lora.alpha = 2 * lora_rank
        config.peft.lora.dropout = 0
        config.peft.lora.target_modules = [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_up_proj",
            "down_proj",
        ]
    tokenizer = SimpleNamespace(vocab_size=32, padded_vocab_size=32, eod_token_id=1, pad_token_id=0)
    monkeypatch.setattr("ironcore.language_model.get_tokenizer", lambda: tokenizer)
    monkeypatch.setattr("ironcore.layers.embedding.get_tokenizer", lambda: tokenizer)
    monkeypatch.setattr(parallel_states, "_TENSOR_MODEL_PARALLEL_WORLD_SIZE", 1)
    monkeypatch.setattr(parallel_states, "_DATA_PARALLEL_WORLD_SIZE", 1)
    monkeypatch.setattr(parallel_states, "get_tensor_model_parallel_rank", lambda: 0)
    monkeypatch.setattr(
        global_vars,
        "GLOBAL_STATES",
        SimpleNamespace(
            timer=Timer(),
            get_logger=lambda: logging.getLogger("granite-tests"),
            get_tokenizer=lambda: tokenizer,
        ),
    )
    native = LanguageModel(config).float()
    weights = WeightMapper(Architecture.GRANITEMOE, 2).hf_to_ironcore(
        {k: v for k, v in reference.state_dict().items() if not (tied and k == "lm_head.weight")}
    )
    if lora:
        weights = {
            key
            if key in native.state_dict()
            else key.replace(".weight", ".base_layer.weight"): value
            for key, value in weights.items()
        }
    info = native.load_state_dict(weights, strict=False)
    assert not info.unexpected_keys
    assert all("lora_" in key or key == "rotary_pos_emb.theta" for key in info.missing_keys)
    freeze_base_model(native, config.peft.method)
    if lora:
        from peft import LoraConfig, get_peft_model

        reference = get_peft_model(
            reference,
            LoraConfig(
                r=lora_rank,
                lora_alpha=2 * lora_rank,
                lora_dropout=0,
                bias="none",
                task_type="CAUSAL_LM",
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
                target_parameters=["gate_up_proj", "down_proj"],
            ),
        )
    return native, reference, config
