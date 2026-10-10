# Copyright (c) 2025-2026 Jaegeun Han
# SPDX-License-Identifier: Apache-2.0
"""HF + PEFT reference, with explicit per-projection adapters for packed experts.

The expert reference uses HF's packed weights and F.linear, without calling
IronCore's expert, routing execution, adapter, or export implementations.
"""

import torch
from torch import nn
from torch.nn import functional as F


class ReferenceLowRank(nn.Module):
    def __init__(self, inputs, outputs, rank, scaling):
        super().__init__()
        self.a = nn.Linear(inputs, rank, bias=False)
        self.b = nn.Linear(rank, outputs, bias=False)
        self.scaling = scaling

    def forward(self, inputs):
        return self.b(self.a(inputs.to(self.a.weight.dtype))) * self.scaling


class ReferenceExperts(nn.Module):
    def __init__(self, base, rank, scaling, targets, accumulation_dtype=None):
        super().__init__()
        self.base = base
        self.accumulation_dtype = accumulation_dtype
        sizes = {
            "gate_proj": (base.hidden_dim, base.intermediate_dim),
            "up_proj": (base.hidden_dim, base.intermediate_dim),
            "down_proj": (base.intermediate_dim, base.hidden_dim),
        }
        self.adapters = nn.ModuleList(
            [
                nn.ModuleDict(
                    {
                        name: ReferenceLowRank(*sizes[name], rank, scaling)
                        for name in targets
                        if name in sizes
                    }
                )
                for _ in range(base.num_experts)
            ]
        )

    def forward(self, hidden_states, top_k_index, top_k_weights):
        result = torch.zeros_like(
            hidden_states, dtype=self.accumulation_dtype or hidden_states.dtype
        )
        for expert in range(self.base.num_experts):
            token, slot = torch.where(top_k_index == expert)
            if token.numel() == 0:
                continue
            inputs = hidden_states[token]
            gate, up = F.linear(inputs, self.base.gate_up_proj[expert]).chunk(2, dim=-1)
            adapters = self.adapters[expert]
            if "gate_proj" in adapters:
                gate = gate + adapters["gate_proj"](inputs)
            if "up_proj" in adapters:
                up = up + adapters["up_proj"](inputs)
            activated = self.base.act_fn(gate) * up
            output = F.linear(activated, self.base.down_proj[expert])
            if "down_proj" in adapters:
                output = output + adapters["down_proj"](activated)
            weighted = output * top_k_weights[token, slot, None]
            result = result.index_add(0, token, weighted.to(result.dtype))
        return result.to(hidden_states.dtype)


def gemma4_peft_reference(native, reference, expert_accumulation_dtype=None):
    """Return an independent HF model and native/transpose-HF parameter pairs."""
    from peft import LoraConfig, get_peft_model

    lora = native.config.peft.lora
    reference = get_peft_model(
        reference,
        LoraConfig(
            r=lora.r,
            lora_alpha=lora.alpha,
            lora_dropout=0,
            target_modules=lora.target_modules,
            bias="none",
            task_type="CAUSAL_LM",
        ),
    )
    pairs = {}
    for layer, (n, h) in enumerate(
        zip(native.model.layers, reference.base_model.model.model.layers, strict=True)
    ):
        for scope in ("self_attn", "mlp"):
            for name in lora.target_modules:
                nm, hm = (
                    getattr(getattr(n, scope), name, None),
                    getattr(getattr(h, scope), name, None),
                )
                if nm is None or not hasattr(nm, "lora"):
                    continue
                key = f"{layer}.{scope}.{name}"
                pairs[key + ".A"] = (nm.lora.lora_A, hm.lora_A["default"].weight)
                pairs[key + ".B"] = (nm.lora.lora_B, hm.lora_B["default"].weight)
        if hasattr(h, "experts"):
            h.experts = ReferenceExperts(
                h.experts, lora.r, lora.scaling, lora.target_modules, expert_accumulation_dtype
            )
            for expert, (ne, he) in enumerate(zip(n.experts, h.experts.adapters, strict=True)):
                for name, adapter in he.items():
                    na = getattr(ne, "lora_" + name)
                    key = f"{layer}.experts.{expert}.{name}"
                    pairs[key + ".A"] = (na.lora_A, adapter.a.weight)
                    pairs[key + ".B"] = (na.lora_B, adapter.b.weight)
    # Shared KV modules can exist in HF but are not called; keep those frozen.
    matched = {id(q) for _, q in pairs.values()}
    for p in reference.parameters():
        p.requires_grad_(id(p) in matched)
    assert {id(p) for p in native.parameters() if p.requires_grad} == {
        id(p) for p, _ in pairs.values()
    }
    return reference, pairs
