# Copyright (c) 2025-2026 Jaegeun Han
#
# SPDX-License-Identifier: Apache-2.0

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from ironcore.config import MainConfig
from ironcore.layers import BaseModule
from ironcore.layers.attention import Attention
from ironcore.layers.layernorm import get_norm
from ironcore.layers.mlp import MLP
from ironcore.parallel.tensor_parallel import ColumnParallelLinear, RowParallelLinear
from ironcore.peft import wrap_with_lora_if_target


class TransformerLayer(BaseModule):
    def __init__(self, config: MainConfig):
        super().__init__(config)

        self.model_config = config.model
        self.init_config = config.init

        # QKV projection dimensions
        query_projection_size = config.model.head_dim * config.model.num_attention_heads
        key_value_projection_size = config.model.head_dim * config.model.num_attention_groups * 2
        self.head_dimension = config.model.d_model // config.model.num_attention_heads

        # tensor parallel attention info
        self.num_local_attention_heads = (
            config.model.num_attention_heads // config.trainer.tensor_model_parallel_size
        )
        self.num_local_attention_groups = (
            config.model.num_attention_groups // config.trainer.tensor_model_parallel_size
        )

        bias_cfg = config.model.bias

        # QKV projections
        self.linear_q = ColumnParallelLinear(
            config,
            config.model.d_model,
            query_projection_size,
            bias=bias_cfg.q,
        )
        kv_has_bias = bias_cfg.k or bias_cfg.v
        self.linear_kv = ColumnParallelLinear(
            config,
            config.model.d_model,
            key_value_projection_size,
            bias=kv_has_bias,
            concatenated_weights=2,
        )

        # If only one of K/V has bias, zero-mask the inactive half of the fused KV bias
        if kv_has_bias and bias_cfg.k != bias_cfg.v:
            local_kv_size = self.linear_kv.bias.shape[0]
            local_half = local_kv_size // 2
            mask = torch.ones(local_kv_size, device=self.linear_kv.bias.device)
            if not bias_cfg.k:
                mask[:local_half] = 0.0  # zero out K portion
            else:
                mask[local_half:] = 0.0  # zero out V portion
            self.register_buffer("_kv_bias_mask", mask, persistent=False)
            with torch.no_grad():
                self.linear_kv.bias.data.mul_(mask)
            self.linear_kv.bias.register_hook(lambda grad: grad * self._kv_bias_mask)

        # Output projection
        self.attn_output = RowParallelLinear(
            config,
            query_projection_size,
            config.model.d_model,
            bias=bias_cfg.o,
            input_is_parallel=True,
        )

        # Wrap with LoRA if PEFT is enabled
        if config.peft.method == "lora":
            self.linear_q = wrap_with_lora_if_target(self.linear_q, "q_proj", config.peft.lora)

            # Handle K and V projections (concatenated layer)
            self.linear_kv = wrap_with_lora_if_target(
                self.linear_kv, ["k_proj", "v_proj"], config.peft.lora, concatenated=True
            )

            self.attn_output = wrap_with_lora_if_target(
                self.attn_output, "o_proj", config.peft.lora
            )

        self.input_layernorm = get_norm(config)
        self.self_attention = Attention(config)
        self.post_attn_layernorm = get_norm(config)

        # Conditional MoE/MLP selection
        if config.model.moe.use_moe:
            from ironcore.layers.moe import MoEMLP

            self.mlp = MoEMLP(config)
        else:
            self.mlp = MLP(config)

        self.residual_dropout = nn.Dropout(config.model.dropout_attn)

    def custom_forward(
        self,
        hidden_states,
        attention_mask,
        rotary_pos_emb,
        position_ids=None,
        use_cache=False,
        past_key_value=None,
        kv_cache_manager=None,
        cache_position=None,
    ):
        # hidden_states: [b, s, h]
        batch_size = hidden_states.size(0)
        seq_len = hidden_states.size(1)

        norm_output = self.input_layernorm(hidden_states)

        # QKV projection
        query = self.linear_q(norm_output)  # [b, sq, hn * hd]
        key_value = self.linear_kv(norm_output)  # [b, sk, 2 * gn * hd]
        key, value = torch.chunk(key_value, 2, dim=-1)  # 2 * [b, sk, gn * hd]

        # reshape to add head dimension
        query = query.view(batch_size, seq_len, self.num_local_attention_heads, self.head_dimension)
        key = key.view(batch_size, seq_len, self.num_local_attention_groups, self.head_dimension)
        value = value.view(
            batch_size, seq_len, self.num_local_attention_groups, self.head_dimension
        )

        # apply rotary positional embedding if provided
        if rotary_pos_emb:
            query = rotary_pos_emb.forward(query, position_ids)
            key = rotary_pos_emb.forward(key, position_ids)

        # KV cache handling:
        # 1. If use_cache=True: return new KV for explicit caching (prefill/generation)
        # 2. If past_key_value provided: use it for attention
        # 3. If kv_cache_manager initialized and not using explicit cache: use manager
        if use_cache or past_key_value is not None:
            # Explicit KV cache path - either generating new KV or using existing
            attn_output = self.self_attention(
                query, key, value, attention_mask, use_cache=use_cache, past_kv=past_key_value
            )
            if use_cache:
                attn_output, new_kv = attn_output
            else:
                new_kv = None
        elif (
            kv_cache_manager is not None
            and cache_position is not None
            and kv_cache_manager.is_initialized
        ):
            # KV cache manager path (for managed inference)
            full_key, full_value = kv_cache_manager.update_layer(
                self.layer_idx, key, value, position=cache_position
            )
            attn_output = self.self_attention(query, full_key, full_value, attention_mask)
            new_kv = None
        else:
            # No caching
            attn_output = self.self_attention(
                query, key, value, attention_mask, use_cache=False, past_kv=None
            )
            new_kv = None

        # output projection
        attention_output = self.attn_output(attn_output)

        if self.config.model.dropout_attn > 0.0:
            attention_output = self.residual_dropout(attention_output)

        residual = hidden_states
        norm_input = residual + attention_output
        norm_output = self.post_attn_layernorm(norm_input)
        mlp_output = self.mlp(norm_output)
        output = norm_input + mlp_output

        if use_cache or kv_cache_manager is not None:
            return output, new_kv
        return output

    def forward(
        self,
        hidden_states,
        attention_mask,
        rotary_pos_emb,
        position_ids=None,
        use_cache=False,
        past_key_value=None,
        kv_cache_manager=None,
        cache_position=None,
    ):
        return self.custom_forward(
            hidden_states,
            attention_mask,
            rotary_pos_emb,
            position_ids=position_ids,
            use_cache=use_cache,
            past_key_value=past_key_value,
            kv_cache_manager=kv_cache_manager,
            cache_position=cache_position,
        )


class TransformerModel(BaseModule):
    def __init__(self, config: MainConfig):
        super().__init__(config)
        self.layers = nn.ModuleList()
        for i in range(config.model.num_layers):
            layer = TransformerLayer(config)
            layer.layer_idx = i
            self.layers.append(layer)

        # Activation checkpointing configuration
        self.activation_recompute = config.operation.activation_recompute
        self.use_reentrant = config.operation.recompute_strategy == "optimized"

    def _is_fsdp_enabled(self) -> bool:
        """Check if this module is being used with FSDP (FSDP wrapper in parent chain)."""
        # Note: Layer-level checkpointing via torch.utils.checkpoint is incompatible
        # with FSDP's parameter sharding. When FSDP is enabled, use FSDP's native
        # apply_activation_checkpointing() instead (applied in parallel.py).
        try:
            import torch.distributed as dist
            if dist.is_initialized():
                # Check if FSDP is likely in use by checking world size > 1
                # and the config specifies FSDP
                return getattr(self.config.parallel, 'use_fsdp', False)
        except (ImportError, AttributeError):
            pass
        return False

    def forward(
        self,
        hidden_states,
        attention_mask,
        rotary_pos_emb,
        position_ids=None,
        use_cache=False,
        past_key_values=None,
        kv_cache_manager=None,
        cache_position=None,
    ):
        new_key_values = [] if use_cache else None

        # Determine if we should use activation checkpointing
        # Note: For DDP, use layer-level checkpointing via torch.utils.checkpoint.
        # For FSDP, skip here and use apply_activation_checkpointing() in parallel.py
        # to avoid "tensor data not allocated" errors with FSDP's parameter sharding.
        is_fsdp = self._is_fsdp_enabled()
        use_layer_checkpointing = (
            self.activation_recompute
            and self.training
            and not use_cache
            and kv_cache_manager is None
            and not is_fsdp  # Skip for FSDP - uses module-level checkpointing instead
        )

        for i, layer in enumerate(self.layers):
            past_kv = past_key_values[i] if past_key_values is not None else None

            if use_layer_checkpointing:
                # Layer-level checkpointing for DDP
                layer_out = checkpoint(
                    layer.custom_forward,
                    hidden_states,
                    attention_mask,
                    rotary_pos_emb,
                    position_ids,
                    False,  # use_cache
                    None,   # past_key_value
                    None,   # kv_cache_manager
                    None,   # cache_position
                    use_reentrant=self.use_reentrant,
                )
            else:
                layer_out = layer(
                    hidden_states,
                    attention_mask,
                    rotary_pos_emb,
                    position_ids=position_ids,
                    use_cache=use_cache,
                    past_key_value=past_kv,
                    kv_cache_manager=kv_cache_manager,
                    cache_position=cache_position,
                )

            if use_cache or kv_cache_manager is not None:
                hidden_states, new_kv = layer_out
                if use_cache:
                    new_key_values.append(new_kv)
            else:
                hidden_states = layer_out

        if use_cache or kv_cache_manager is not None:
            return hidden_states, new_key_values
        return hidden_states
