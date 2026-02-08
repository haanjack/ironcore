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
        key_value_projection_size = config.model.head_dim * config.model.num_attention_groups * 2
        self.head_dimension = config.model.d_model // config.model.num_attention_heads

        # tensor parallel attention info
        self.num_local_attention_heads = (
            config.model.num_attention_heads // config.trainer.tensor_model_parallel_size
        )
        self.num_local_attention_groups = (
            config.model.num_attention_groups // config.trainer.tensor_model_parallel_size
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
        value = value.view(
            batch_size, seq_len, self.num_local_attention_groups, self.head_dimension
        )

        # apply rotary positional embedding if provided
        if rotary_pos_emb:
            query = rotary_pos_emb.forward(query)
            key = rotary_pos_emb.forward(key)

        # Chunking for Async TP
        chunk_size = self.config.trainer.sequence_chunk_size

        if chunk_size is None or chunk_size <= 0 or chunk_size >= seq_len:
            # Standard synchronous execution
            # self attention
            attention_output = self.self_attention(query, key, value, attention_mask)

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

        # Use torch.split for splitting
        query_chunks = torch.split(query, chunk_size, dim=1)

        if self.model_config.post_ln:
            residual_base = norm_output
        else:
            residual_base = hidden_states
        residual_chunks = torch.split(residual_base, chunk_size, dim=1)

        attn_partials = []
        attn_handles = []

        # 1. Launch Attention Compute & Reduce for all chunks
        current_idx = 0
        for i, query_chunk in enumerate(query_chunks):
            chunk_len = query_chunk.size(1)
            kv_end = current_idx + chunk_len

            # Truncate key/value to the causal boundary: positions beyond
            # kv_end are masked by causal attention anyway. This ensures
            # correct causal mask alignment with flash attention, which
            # applies bottom-right aligned masking when seq_len_q != seq_len_kv.
            key_chunk = key[:, :kv_end]
            value_chunk = value[:, :kv_end]

            mask_chunk = None
            if attention_mask is not None:
                mask_chunk = attention_mask[:, :, current_idx:kv_end, :kv_end]

            attention_output_chunk = self.self_attention(
                query_chunk, key_chunk, value_chunk, mask_chunk
            )

            # output projection (Async)
            partial, handle = self.attn_output(attention_output_chunk, async_communication=True)
            attn_partials.append(partial)
            attn_handles.append(handle)
            current_idx += chunk_len

        # 2. Finish Attention, Launch MLP
        mlp_partials = []
        mlp_handles = []
        mlp_residual_bases = []

        for i in range(len(query_chunks)):
            # Wait for Attention Reduce
            if attn_handles[i]:
                attn_handles[i].wait()

            # Finish Attention (Bias + Dropout)
            attention_output = attn_partials[i]
            if self.attn_output.bias is not None:
                attention_output = attention_output + self.attn_output.bias

            if self.config.model.dropout_attn > 0.0:
                attention_output = self.residual_dropout(attention_output)

            # Residual + Norm
            norm_input = residual_chunks[i] + attention_output
            norm_output_chunk = self.post_attn_layernorm(norm_input)

            # Store residual base for MLP
            if self.model_config.post_ln:
                mlp_residual_bases.append(norm_output_chunk)
            else:
                mlp_residual_bases.append(norm_input)

            # MLP (Async)
            partial, handle = self.mlp(norm_output_chunk, async_communication=True)
            mlp_partials.append(partial)
            mlp_handles.append(handle)

        # 3. Finish MLP
        final_chunks = []
        for i in range(len(query_chunks)):
            # Wait MLP and Finalize
            mlp_output = self.mlp.finalize(mlp_partials[i], mlp_handles[i])

            output_chunk = mlp_residual_bases[i] + mlp_output
            final_chunks.append(output_chunk)

        return torch.cat(final_chunks, dim=1)

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
                hidden_states = layer(hidden_states, attention_mask, rotary_pos_emb)
        return hidden_states
