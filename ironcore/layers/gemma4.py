# Copyright (c) 2025-2026 Jaegeun Han
#
# SPDX-License-Identifier: Apache-2.0

"""Gemma 4 dense decoder blocks, used by IronCore's TransformerModel.

Reference: transformers/models/gemma4/modeling_gemma4.py. Projection matrices
retain IronCore's [input, output] layout for native checkpoints and LoRA.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from ironcore.config import MainConfig
from ironcore.layers.blockwise import token_chunk_forward
from ironcore.layers.module import BaseModule
from ironcore.parallel.random import checkpoint_with_tensor_parallel_rng
from ironcore.parallel.tensor_parallel import (
    ColumnParallelLinear,
    RowParallelLinear,
    VocabParallelEmbedding,
    comm,
)
from ironcore.peft import wrap_with_lora_if_target

KeyValue = tuple[torch.Tensor, torch.Tensor]


class Gemma4PerLayerEmbedding(VocabParallelEmbedding):
    """Vocabulary-sharded PLE lookup with a single frozen padding row."""

    def __init__(self, config: MainConfig) -> None:
        gemma = config.model.gemma4
        super().__init__(
            config,
            gemma.vocab_size_per_layer_input,
            config.model.num_layers * gemma.hidden_size_per_layer_input,
        )
        local_index = gemma.pad_token_id - self.tensor_model_parallel_rank * self.parallel_input_dim
        self.local_padding_idx = local_index if 0 <= local_index < self.parallel_input_dim else None
        self.weight.register_hook(self._zero_padding_grad)

    def _zero_padding_grad(self, grad: torch.Tensor) -> torch.Tensor:
        if self.local_padding_idx is not None:
            grad[self.local_padding_idx].zero_()
        return grad


class Gemma4RMSNorm(nn.Module):
    """Normalize in FP32, multiply the direct scale, then restore input dtype."""

    def __init__(
        self, dim: int, eps: float = 1e-6, with_scale: bool = True, sum_tp_grad: bool = False
    ) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim)) if with_scale else None
        self.sum_tp_grad = sum_tp_grad

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        values = hidden_states.float()
        values = values * (values.square().mean(-1, keepdim=True) + self.eps).pow(-0.5)
        if self.weight is not None:
            weight = self.weight
            if self.sum_tp_grad:
                weight = comm.copy_inputs_to_model_parallel_workers(weight)
            values = values * weight.float()
        return values.to(hidden_states.dtype)


def _projection(
    config: MainConfig, in_features: int, out_features: int, name: str, row: bool = False
) -> nn.Module:
    if row:
        layer = RowParallelLinear(
            config, in_features, out_features, bias=False, input_is_parallel=True
        )
    else:
        layer = ColumnParallelLinear(config, in_features, out_features, bias=False)
    if config.peft.method == "lora":
        layer = wrap_with_lora_if_target(layer, name, config.peft.lora)
    return layer


class Gemma4Attention(BaseModule):
    """Q/K/V normalization, hybrid attention, and cross-layer KV sharing."""

    def __init__(self, config: MainConfig, layer_idx: int) -> None:
        super().__init__(config)
        model, gemma = config.model, config.model.gemma4
        self.layer_type = gemma.layer_types[layer_idx]
        self.head_dim, self.kv_heads = gemma.head_layout(model, layer_idx)
        tp_size = config.trainer.tensor_model_parallel_size
        self.query_heads = model.num_attention_heads // tp_size
        self.replicate_kv = self.kv_heads < tp_size
        global_kv_heads = self.kv_heads
        if not self.replicate_kv:
            self.kv_heads //= tp_size
        self.is_shared = layer_idx >= model.num_layers - gemma.num_kv_shared_layers
        self.k_eq_v = gemma.attention_k_eq_v and self.layer_type == "full_attention"
        self.sliding_window = (
            gemma.sliding_window if self.layer_type == "sliding_attention" else None
        )
        self.q_proj = _projection(
            config, model.d_model, model.num_attention_heads * self.head_dim, "q_proj"
        )
        self.o_proj = _projection(
            config, model.num_attention_heads * self.head_dim, model.d_model, "o_proj", row=True
        )
        self.q_norm = Gemma4RMSNorm(self.head_dim, model.ln_eps, sum_tp_grad=tp_size > 1)
        if not self.is_shared:
            self.k_proj = _projection(
                config, model.d_model, global_kv_heads * self.head_dim, "k_proj"
            )
            self.v_proj = (
                None
                if self.k_eq_v
                else _projection(config, model.d_model, global_kv_heads * self.head_dim, "v_proj")
            )
            self.k_norm = Gemma4RMSNorm(
                self.head_dim, model.ln_eps, sum_tp_grad=tp_size > 1 and not self.replicate_kv
            )
            self.v_norm = Gemma4RMSNorm(self.head_dim, model.ln_eps, with_scale=False)

        global_attention = self.layer_type == "full_attention"
        theta = gemma.global_rope_theta if global_attention else gemma.sliding_rope_theta
        count = (
            int(self.head_dim * gemma.global_partial_rotary_factor // 2)
            if global_attention
            else self.head_dim // 2
        )
        # p-RoPE keeps the full head layout: unused frequencies are zero, not removed.
        frequencies = torch.zeros(self.head_dim // 2, dtype=torch.float32)
        frequencies[:count] = theta ** (-torch.arange(0, count * 2, 2).float() / self.head_dim)
        self.register_buffer("inv_freq", frequencies, persistent=False)

    def _rotate(self, values: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        with torch.autocast(device_type=values.device.type, enabled=False):
            angles = positions.float().unsqueeze(-1) * self.inv_freq.float()
            angles = torch.cat((angles, angles), dim=-1).unsqueeze(2)
            cos, sin = angles.cos().to(values.dtype), angles.sin().to(values.dtype)
        left, right = values.chunk(2, dim=-1)
        return values * cos + torch.cat((-right, left), dim=-1) * sin

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None,
        position_ids: torch.Tensor,
        past_key_value: KeyValue | None = None,
        shared_key_value: KeyValue | None = None,
    ) -> tuple[torch.Tensor, KeyValue]:
        batch, length = hidden_states.shape[:2]
        query = self.q_proj(hidden_states).view(batch, length, self.query_heads, self.head_dim)
        query = self._rotate(self.q_norm(query), position_ids)
        if self.is_shared:
            if shared_key_value is None:
                raise ValueError("Gemma 4 shared attention requires its producer's KV states")
            key, value = shared_key_value
        else:
            key = self.k_proj(hidden_states)
            value = key if self.v_proj is None else self.v_proj(hidden_states)
            if self.replicate_kv:
                # Shard projection channels, then materialize the complete shared
                # KV head on each rank. Gather after LoRA so adapters follow too.
                attrs = {"column_parallel": True}
                key = comm.gather_from_model_parallel_workers(key, attrs)
                value = (
                    key
                    if self.v_proj is None
                    else comm.gather_from_model_parallel_workers(value, attrs)
                )
            key = key.view(batch, length, self.kv_heads, self.head_dim)
            value = value.view(batch, length, self.kv_heads, self.head_dim)
            value = self.v_norm(value)
            key = self._rotate(self.k_norm(key), position_ids)
            if self.replicate_kv:
                # Consumers compute different query heads. Sum their gradients
                # before normalization and the projection gather split backward.
                key = comm.copy_inputs_to_model_parallel_workers(key)
                value = comm.copy_inputs_to_model_parallel_workers(value)
            if past_key_value is not None:
                key = torch.cat((past_key_value[0], key), dim=1)
                value = torch.cat((past_key_value[1], value), dim=1)
        kv = key, value
        from ironcore.parallel import parallel_states

        if parallel_states.get_context_parallel_world_size() > 1:
            from ironcore.parallel.context_parallel import gather_context_parallel

            key = gather_context_parallel(key, reduce_backward=True)
            value = gather_context_parallel(value, reduce_backward=True)
        if self.config.model.gemma4.attention_chunk_size is not None:
            offset = position_ids[0, 0].item()
            return self._tiled_attention(query, key, value, attention_mask, offset), kv
        key_length = key.size(1)
        queries = torch.arange(key_length - length, key_length, device=query.device).unsqueeze(-1)
        keys = torch.arange(key_length, device=query.device).unsqueeze(0)
        mask = keys <= queries
        if self.sliding_window is not None:
            mask = mask & (keys > queries - self.sliding_window)
        if attention_mask is not None:
            mask = mask & attention_mask.to(device=query.device, dtype=torch.bool)
        repeats = self.query_heads // self.kv_heads
        key = key.transpose(1, 2).repeat_interleave(repeats, dim=1)
        value = value.transpose(1, 2).repeat_interleave(repeats, dim=1)
        attended = F.scaled_dot_product_attention(
            query.transpose(1, 2),
            key,
            value,
            attn_mask=mask,
            dropout_p=self.config.model.dropout_attn if self.training else 0.0,
            scale=1.0,
        )
        attended = attended.transpose(1, 2).reshape(batch, length, -1)
        return self.o_proj(attended), kv

    def _tiled_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: torch.Tensor | None,
        offset: int,
    ) -> torch.Tensor:
        """Bound score memory for 512-dimensional global heads and local windows."""
        length = query.size(1)
        past = offset
        chunk = self.config.model.gemma4.attention_chunk_size
        outputs = []
        for start in range(0, length, chunk):
            end = min(length, start + chunk)
            lo = max(0, past + start - self.sliding_window + 1) if self.sliding_window else 0
            hi = past + end
            mask = None if attention_mask is None else attention_mask[..., start:end, lo:hi]

            def attend(q, k, v, mask, start=start, lo=lo):
                positions = torch.arange(past + start, past + start + q.size(1), device=q.device)
                keys = torch.arange(lo, lo + k.size(1), device=q.device)
                allowed = keys[None] <= positions[:, None]
                if self.sliding_window is not None:
                    allowed = allowed & (keys[None] > positions[:, None] - self.sliding_window)
                if mask is not None:
                    allowed = allowed & mask
                repeats = self.query_heads // self.kv_heads
                k = k.transpose(1, 2).repeat_interleave(repeats, dim=1)
                v = v.transpose(1, 2).repeat_interleave(repeats, dim=1)
                return F.scaled_dot_product_attention(
                    q.transpose(1, 2),
                    k,
                    v,
                    attn_mask=allowed,
                    dropout_p=self.config.model.dropout_attn if self.training else 0.0,
                    scale=1.0,
                ).transpose(1, 2)

            args = query[:, start:end], key[:, lo:hi], value[:, lo:hi], mask
            out = (
                checkpoint_with_tensor_parallel_rng(attend, *args, use_reentrant=False)
                if self.training and torch.is_grad_enabled()
                else attend(*args)
            )
            outputs.append(out)
        attended = torch.cat(outputs, dim=1).reshape(query.size(0), length, -1)
        return self.o_proj(attended)


class Gemma4MLP(BaseModule):
    """GELU-tanh gated MLP, including E2B's double-wide shared layers."""

    def __init__(self, config: MainConfig, layer_idx: int) -> None:
        super().__init__(config)
        model, gemma = config.model, config.model.gemma4
        shared = layer_idx >= model.num_layers - gemma.num_kv_shared_layers
        width = model.d_ffn * (2 if shared and gemma.use_double_wide_mlp else 1)
        self.gate_proj = _projection(config, model.d_model, width, "gate_proj")
        self.up_proj = _projection(config, model.d_model, width, "up_proj")
        self.down_proj = _projection(config, width, model.d_model, "down_proj", row=True)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return token_chunk_forward(
            self._project, hidden_states, self.config.trainer.mlp_chunk_size, training=self.training
        )

    def _project(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.down_proj(
            F.gelu(self.gate_proj(hidden_states), approximate="tanh") * self.up_proj(hidden_states)
        )


class Gemma4Router(nn.Module):
    """Reference RMS-scaled routing with normalized top-k and expert scales."""

    def __init__(self, config: MainConfig) -> None:
        super().__init__()
        hidden = config.model.d_model
        self.norm = Gemma4RMSNorm(hidden, config.model.ln_eps, with_scale=False)
        self.proj = nn.Linear(hidden, config.model.moe.num_routed_experts, bias=False)
        self.scale = nn.Parameter(torch.ones(hidden))
        self.per_expert_scale = nn.Parameter(torch.ones(config.model.moe.num_routed_experts))
        self.top_k = config.model.moe.num_experts_per_token
        self.scalar_root_size = hidden**-0.5

    def forward(self, hidden: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        scores = self.proj(self.norm(hidden) * self.scale * self.scalar_root_size)
        probabilities = scores.float().softmax(-1)
        weights, indices = probabilities.topk(self.top_k, dim=-1)
        weights = weights / weights.sum(-1, keepdim=True)
        return indices, weights * self.per_expert_scale[indices]


class Gemma4GatedActivation(nn.Module):
    """GELU-tanh on the first projection half times the second half."""

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        gate, up = value.chunk(2, dim=-1)
        return F.gelu(gate, approximate="tanh") * up


class Gemma4Expert(BaseModule):
    """Fused gate/up expert compatible with the bounded grouped GEMM backend."""

    def __init__(self, config: MainConfig) -> None:
        super().__init__(config)
        width = config.model.moe.expert_intermediate_size
        self.up_proj = ColumnParallelLinear(
            config, config.model.d_model, 2 * width, bias=False, concatenated_weights=2
        )
        self.down_proj = RowParallelLinear(
            config, width, config.model.d_model, bias=False, input_is_parallel=True
        )
        self.activation = Gemma4GatedActivation()

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.activation(self.up_proj(value)))


class Gemma4Layer(BaseModule):
    """Four-normalization decoder block with optional gated PLE injection."""

    def __init__(self, config: MainConfig, layer_idx: int) -> None:
        super().__init__(config)
        self.layer_idx = layer_idx
        model, gemma = config.model, config.model.gemma4
        self.self_attn = Gemma4Attention(config, layer_idx)
        self.mlp = Gemma4MLP(config, layer_idx)
        self.input_layernorm = Gemma4RMSNorm(model.d_model, model.ln_eps)
        self.post_attention_layernorm = Gemma4RMSNorm(model.d_model, model.ln_eps)
        self.pre_feedforward_layernorm = Gemma4RMSNorm(model.d_model, model.ln_eps)
        self.post_feedforward_layernorm = Gemma4RMSNorm(model.d_model, model.ln_eps)
        if model.moe.use_moe:
            self.router = Gemma4Router(config)
            self.experts = nn.ModuleList(
                [Gemma4Expert(config) for _ in range(model.moe.num_routed_experts)]
            )
            self.post_feedforward_layernorm_1 = Gemma4RMSNorm(model.d_model, model.ln_eps)
            self.post_feedforward_layernorm_2 = Gemma4RMSNorm(model.d_model, model.ln_eps)
            self.pre_feedforward_layernorm_2 = Gemma4RMSNorm(model.d_model, model.ln_eps)
        self.register_buffer("layer_scalar", torch.ones(1))
        if gemma.hidden_size_per_layer_input:
            self.per_layer_input_gate = _projection(
                config, model.d_model, gemma.hidden_size_per_layer_input, "per_layer_input_gate"
            )
            self.per_layer_projection = _projection(
                config,
                gemma.hidden_size_per_layer_input,
                model.d_model,
                "per_layer_projection",
                row=True,
            )
            self.post_per_layer_input_norm = Gemma4RMSNorm(model.d_model, model.ln_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None,
        position_ids: torch.Tensor,
        past_key_value: KeyValue | None = None,
        shared_key_value: KeyValue | None = None,
        per_layer_input: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, KeyValue]:
        attention, kv = self.self_attn(
            self.input_layernorm(hidden_states),
            attention_mask,
            position_ids,
            past_key_value,
            shared_key_value,
        )
        hidden_states = hidden_states + self.post_attention_layernorm(attention)
        mlp = self.mlp(self.pre_feedforward_layernorm(hidden_states))
        if self.config.model.moe.use_moe:
            flat = hidden_states.reshape(-1, hidden_states.size(-1))
            indices, weights = self.router(flat)
            expert_inputs = self.pre_feedforward_layernorm_2(flat)
            if self.config.model.moe.expert_backend == "grouped":
                from ironcore.layers.moe.grouped import grouped_experts

                routed = grouped_experts(expert_inputs, indices, weights, self.experts)
            else:
                routed = torch.zeros_like(flat)
                for index, expert in enumerate(self.experts):
                    tokens, slots = torch.where(indices == index)
                    values = expert(expert_inputs[tokens]) * weights[tokens, slots, None]
                    routed = routed.index_add(0, tokens, values.to(routed.dtype))
            mlp = self.post_feedforward_layernorm_1(mlp) + self.post_feedforward_layernorm_2(
                routed.to(flat.dtype).reshape_as(hidden_states)
            )
        hidden_states = hidden_states + self.post_feedforward_layernorm(mlp)
        if self.config.model.gemma4.hidden_size_per_layer_input:
            if per_layer_input is None:
                raise ValueError("Gemma 4 PLE-enabled layers require per-layer inputs")
            per_layer_input = comm.scatter_input_to_model_parallel_workers(per_layer_input)
            gated = F.gelu(self.per_layer_input_gate(hidden_states), approximate="tanh")
            hidden_states = hidden_states + self.post_per_layer_input_norm(
                self.per_layer_projection(gated * per_layer_input)
            )
        return hidden_states * self.layer_scalar, kv

    def forward_checkpointed(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None,
        position_ids: torch.Tensor,
        shared_key: torch.Tensor | None,
        shared_value: torch.Tensor | None,
        per_layer_input: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Expose both inputs and outputs individually to reentrant autograd."""
        shared = (shared_key, shared_value) if shared_key is not None else None
        output, (key, value) = self(
            hidden_states, attention_mask, position_ids, None, shared, per_layer_input
        )
        return output, key, value

    def forward_spilled(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None,
        position_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Full-layer tensor output for scheduler-controlled CPU activation spilling."""
        return self.forward(hidden_states, attention_mask, position_ids)[0]
