# Copyright (c) 2025-2026 Jaegeun Han
# SPDX-License-Identifier: Apache-2.0
"""Granite packed-parameter LoRA and model-dtype router projections."""

import torch
import torch.nn.functional as F

from ironcore.parallel.tensor_parallel import comm
from ironcore.peft.lora import LoRALinear

from .expert import ExpertMLP
from .router import TopKRouter


class GraniteRouter(TopKRouter):
    def _compute_router_logits(self, hidden_states):
        # HF projects in model/autocast dtype, then selects/normalizes in FP32.
        return F.linear(hidden_states, self.weight.T).float()


class GraniteExpert(ExpertMLP):
    """A shared gate/up A factor, matching HF PEFT's 3D parameter LoRA.

    Expert factors are replicated FP32 leaves; temporary TP views are folded
    directly into BF16 base weights with baddbmm. This is parameter LoRA, so
    activation dropout is unsupported. The base weights remain frozen.
    """

    def __init__(self, config, hidden_size, intermediate_size, expert_id=0):
        super().__init__(config, hidden_size, intermediate_size, expert_id)
        self.folded_parameter_lora = True
        self.activation.hf_weight_layout = True
        if config.peft.method == "lora":
            lora = config.peft.lora
            for name, inputs, outputs in (
                ("gate_up_proj", hidden_size, 2 * intermediate_size),
                ("down_proj", intermediate_size, hidden_size),
            ):
                if name in lora.target_modules:
                    self.add_module(
                        f"lora_{name}",
                        LoRALinear(inputs, outputs, lora.r, lora.alpha, lora.dropout),
                    )

    def forward(self, value, async_communication=False, **kwargs):
        from .lora import expert_parameters

        weights, _, _ = expert_parameters([self], [0])
        value = comm.copy_inputs_to_model_parallel_workers(value)
        gate, up = F.linear(value, weights[0].T).chunk(2, dim=-1)
        output = F.linear(F.silu(gate) * up, weights[1].T)
        # Keep the loop backend's async return contract. The bounded grouped
        # path performs its own batched TP reduction outside this method.
        output = comm.reduce_inputs_from_model_parallel_workers(output)
        return (output, None) if async_communication else output


def folded_expert_parameters(experts, active):
    """Pack adapter communication outside group recomputation, then fold.

    baddbmm writes model-dtype weights directly, without an FP32 full-weight
    result subsequently cast to BF16. Its GEMM workspace is backend-owned.
    """
    selected = [experts[i] for i in active]
    weights = []
    for name, base in (("gate_up_proj", "up_proj"), ("down_proj", "down_proj")):
        weight = torch.stack([getattr(e, base).weight for e in selected])
        adapters = [getattr(e, f"lora_{name}", None) for e in selected]
        if adapters[0] is not None:
            a = torch.stack([adapter.lora_A for adapter in adapters])
            b = torch.stack([adapter.lora_B for adapter in adapters])
            if name == "down_proj":
                a = comm.scatter_input_to_model_parallel_workers(a.transpose(-1, -2)).transpose(
                    -1, -2
                )
                b = comm.copy_inputs_to_model_parallel_workers(b)
            else:
                a = comm.copy_inputs_to_model_parallel_workers(a)
                gate, up = b.chunk(2, dim=-1)
                b = torch.cat(
                    [comm.scatter_input_to_model_parallel_workers(part) for part in (gate, up)],
                    dim=-1,
                )
            with torch.autocast(device_type=weight.device.type, enabled=False):
                # Use HF's [expert, output, input] multiplication orientation.
                weight = torch.baddbmm(
                    weight.transpose(-1, -2).contiguous(),
                    # PEFT stores packed B as [output, rank, expert], then
                    # permutes it to [expert, output, rank]. Match its strides
                    # as well as its values, including nonzero trained B.
                    b.permute(2, 1, 0).contiguous().permute(2, 0, 1).to(weight.dtype),
                    a.transpose(-1, -2).contiguous().to(weight.dtype),
                    alpha=adapters[0].scaling,
                ).transpose(-1, -2)
        weights.append(weight)
    return tuple(p for row in zip(*weights, strict=True) for p in row), (), 2
