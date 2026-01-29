# Copyright (c) 2025-2026 Jaegeun Han
#
# SPDX-License-Identifier: MIT
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the above copyright notice,
# this list of conditions, and the following disclaimer are retained.
#
# Full license text is available at LICENSE file.

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from ironcore.config import MainConfig
from ironcore.layers import BaseModule
from ironcore.layers.attention import Attention
from ironcore.layers.layernorm import get_norm
from ironcore.layers.mlp import MLP
from ironcore.parallel.tensor_parallel import ColumnParallelLinear, RowParallelLinear


class TransformerLayer(BaseModule):
    def __init__(self, config: MainConfig):
        super().__init__(config)

        self.model_config = config.model
        self.init_config = config.init

        # QKV projection dimensions
        query_projection_size = config.model.head_dim * config.model.num_attention_heads
        key_value_projection_size = (
            config.model.head_dim * config.model.num_attention_groups * 2
        )
        self.head_dimension = config.model.d_model // config.model.num_attention_heads

        # tensor parallel attention info
        self.num_local_attention_heads = (
            config.model.num_attention_heads // config.trainer.tensor_model_parallel_size
        )
        self.num_local_attention_groups = (
            config.model.num_attention_groups
            // config.trainer.tensor_model_parallel_size
        )

        # QKV projections
        self.linear_q = ColumnParallelLinear(
            config,
            config.model.d_model,
            query_projection_size,
            bias=not config.model.no_bias,
        )
        self.linear_kv = ColumnParallelLinear(
            config,
            config.model.d_model,
            key_value_projection_size,
            bias=not config.model.no_bias,
            concatenated_weights=2,
        )

        # Output projection
        self.attn_output = RowParallelLinear(
            config,
            query_projection_size,
            config.model.d_model,
            bias=not config.model.no_bias,
            input_is_parallel=True,
        )

        self.input_layernorm = get_norm(config)
        self.self_attention = Attention(config)
        self.post_attn_layernorm = get_norm(config)
        self.mlp = MLP(config)

        self.residual_dropout = nn.Dropout(config.model.dropout_attn)

    def custom_forward(self, hidden_states, attention_mask, rotary_pos_emb):

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
        value = value.view(batch_size, seq_len, self.num_local_attention_groups, self.head_dimension)

        # apply rotary positional embedding if provided
        if rotary_pos_emb:
            query = rotary_pos_emb.forward(query)
            key = rotary_pos_emb.forward(key)

        # self attention
        attention_output = self.self_attention(
            query, key, value, attention_mask
        )

        # output projection
        attention_output = self.attn_output(attention_output)

        # dropout
        if self.config.model.dropout_attn > 0.0:
            attention_output = self.residual_dropout(attention_output)

        if self.model_config.post_ln:
            residual = norm_output
        else:
            residual = hidden_states

        # dropout
        norm_input = residual + attention_output

        # layer norm after attention
        norm_output = self.post_attn_layernorm(norm_input)

        mlp_output = self.mlp(norm_output)

        if self.model_config.post_ln:
            residual = norm_output
        else:
            residual = norm_input

        output = residual + mlp_output

        return output

    def forward(self, hidden_states, attention_mask, rotary_pos_emb):
        return self.custom_forward(hidden_states, attention_mask, rotary_pos_emb)


class TransformerModel(BaseModule):

    def __init__(self, config: MainConfig):

        super().__init__(config)

        # TODO: create layers considering the number of layers per pipeline parallel group size
        self.layers = nn.ModuleList(
            [TransformerLayer(config) for _ in range(config.model.num_layers)]
        )

    def forward(self, hidden_states, attention_mask, rotary_pos_emb):
        for layer in self.layers:
            if self.config.operation.activation_recompute:
                hidden_states = checkpoint(
                    layer.custom_forward,
                    hidden_states,
                    attention_mask,
                    rotary_pos_emb,
                    use_reentrant=self.use_reentrant,
                )
            else:
                hidden_states = layer(
                    hidden_states, attention_mask, rotary_pos_emb)
        return hidden_states
