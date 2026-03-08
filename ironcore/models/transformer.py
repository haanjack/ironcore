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

        # Prepare past_kv for attention (will be used for concatenation inside attention)
        past_kv_for_attn = past_key_value if use_cache else None

        # Chunking for Async TP
        chunk_size = self.config.trainer.sequence_chunk_size

        if chunk_size is None or chunk_size <= 0 or chunk_size >= seq_len:
            # Standard synchronous execution
            # self attention
            attn_output = self.self_attention(
                query, key, value, attention_mask, use_cache=use_cache, past_kv=past_kv_for_attn
            )

            # Handle cache return
            if use_cache:
                attention_output, new_kv = attn_output
            else:
                attention_output = attn_output

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

            if use_cache:
                return output, new_kv
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
        final_new_kv = None  # Will store KV from last chunk if using cache

        # 1. Launch Attention Compute & Reduce for all chunks
        current_idx = 0
        num_chunks = len(query_chunks)

        for i, query_chunk in enumerate(query_chunks):
            chunk_len = query_chunk.size(1)

            # For chunked async TP, each query chunk must attend to:
            # 1. All past cached KV (from previous forward passes)
            # 2. ALL new KV from this forward pass (not just up to current chunk)
            #
            # The attention_mask passed in has shape [batch, 1, seq_len, total_len] where:
            # - seq_len: length of new tokens in this forward pass
            # - total_len: cache_position + seq_len (full context including cached tokens)
            #
            # When we slice the mask for a chunk:
            # - mask_chunk = attention_mask[:, :, current_idx:(current_idx + chunk_len), :]
            # - This gives us [batch, 1, chunk_len, total_len]
            # - The last dimension (total_len) correctly covers all KV positions
            #
            # Inside self_attention, when past_kv is provided:
            # - key/value are concatenated with cached KV: [batch, total_len, gn, hd]
            # - The mask's last dimension matches this concatenated KV length
            # - Causal masking ensures each query position only attends to valid previous tokens
            mask_chunk = None
            if attention_mask is not None:
                # Extract the relevant portion of the attention mask for this query chunk
                # Note: This slices the query dimension (dim 2), keeping full KV dimension (dim 3)
                mask_chunk = attention_mask[:, :, current_idx : (current_idx + chunk_len), :]

            # Pass full new KV (not chunked) so each query chunk can attend to all new tokens
            # The causal mask will ensure proper attention boundaries
            attn_out = self.self_attention(
                query_chunk, key, value, mask_chunk, use_cache=use_cache, past_kv=past_kv_for_attn
            )

            # Handle cache in chunked path
            if use_cache:
                attention_output_chunk, chunk_kv = attn_out
                # For chunked execution with cache, we use the last chunk's KV
                # which contains past_kv + all new KV
                if i == num_chunks - 1:
                    final_new_kv = chunk_kv
            else:
                attention_output_chunk = attn_out

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
            # Finish Attention (handle LoRA if present)
            # Check if attn_output has finalize method (LoRA-wrapped)
            if hasattr(self.attn_output, "finalize"):
                attention_output = self.attn_output.finalize(attn_partials[i], attn_handles[i])
            else:
                # Standard path without LoRA
                if attn_handles[i]:
                    attn_handles[i].wait()

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

        output = torch.cat(final_chunks, dim=1)
        if use_cache:
            # final_new_kv should be set from the last chunk
            assert final_new_kv is not None, "Cache enabled but no KV was captured"
            return output, final_new_kv
        return output

    def forward(
        self,
        hidden_states,
        attention_mask,
        rotary_pos_emb,
        position_ids=None,
        use_cache=False,
        past_key_value=None,
    ):
        return self.custom_forward(
            hidden_states,
            attention_mask,
            rotary_pos_emb,
            position_ids=position_ids,
            use_cache=use_cache,
            past_key_value=past_key_value,
        )

    def forward_with_cache(
        self,
        hidden_states,
        attention_mask,
        rotary_pos_emb,
        kv_cache_manager,
        layer_idx,
        cache_position,
        position_ids=None,
    ):
        """Forward pass using stateful KVCacheManager.

        Args:
            hidden_states: [b, s, h]
            attention_mask: [b, 1, s, s]
            rotary_pos_emb: Rotary position embedding
            kv_cache_manager: KVCacheManager instance for stateful caching
            layer_idx: Layer index (0-indexed)
            cache_position: Starting position in cache
            position_ids: Optional position IDs [b, s]

        Returns:
            Output tensor [b, s, h]
        """
        batch_size, seq_len = hidden_states.shape[:2]

        norm_output = self.input_layernorm(hidden_states)

        # QKV projection
        query = self.linear_q(norm_output)
        key_value = self.linear_kv(norm_output)
        key, value = torch.chunk(key_value, 2, dim=-1)

        # Reshape for attention
        query = query.view(batch_size, seq_len, self.num_local_attention_heads, self.head_dimension)
        key = key.view(batch_size, seq_len, self.num_local_attention_groups, self.head_dimension)
        value = value.view(batch_size, seq_len, self.num_local_attention_groups, self.head_dimension)

        # Apply RoPE
        if rotary_pos_emb:
            query = rotary_pos_emb.forward(query, position_ids)
            key = rotary_pos_emb.forward(key, position_ids)

        # Update cache and get full KV
        full_key, full_value = kv_cache_manager.update_layer(
            layer_idx=layer_idx,
            key=key,
            value=value,
            position=cache_position,
        )

        # Attention with full cached KV
        attention_output = self.self_attention(
            query, full_key, full_value, attention_mask, use_cache=False, past_kv=None
        )

        # Output projection
        attention_output = self.attn_output(attention_output)

        # Dropout
        if self.config.model.dropout_attn > 0.0:
            attention_output = self.residual_dropout(attention_output)

        # Residual connection
        if self.model_config.post_ln:
            residual = norm_output
        else:
            residual = hidden_states
        norm_input = residual + attention_output

        # Layer norm after attention
        norm_output = self.post_attn_layernorm(norm_input)

        # MLP
        mlp_output = self.mlp(norm_output)

        if self.model_config.post_ln:
            residual = norm_output
        else:
            residual = norm_input

        output = residual + mlp_output

        return output


class TransformerModel(BaseModule):
    def __init__(self, config: MainConfig):

        super().__init__(config)

        # TODO: create layers considering the number of layers per pipeline parallel group size
        self.layers = nn.ModuleList(
            [TransformerLayer(config) for _ in range(config.model.num_layers)]
        )

    def forward(
        self,
        hidden_states,
        attention_mask,
        rotary_pos_emb,
        position_ids=None,
        use_cache=False,
        past_key_values=None,
    ):
        """
        Forward pass through all transformer layers.

        Args:
            hidden_states: [b, s, h]
            attention_mask: [b, 1, s, s]
            rotary_pos_emb: Rotary position embedding
            position_ids: Optional position IDs [b, s]
            use_cache: Whether to use KV cache
            past_key_values: List of past (key, value) tuples for each layer

        Returns:
            If use_cache: (hidden_states, new_key_values)
            Otherwise: hidden_states
        """
        new_key_values = [] if use_cache else None

        for i, layer in enumerate(self.layers):
            past_kv = past_key_values[i] if past_key_values is not None else None

            if self.config.operation.activation_recompute:
                # Note: Gradient checkpointing with cache is complex
                # For now, disable cache with activation recompute
                if use_cache:
                    raise NotImplementedError(
                        "KV cache with activation_recompute not yet supported"
                    )
                hidden_states = checkpoint(
                    layer.custom_forward,
                    hidden_states,
                    attention_mask,
                    rotary_pos_emb,
                    position_ids,
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
                )
                if use_cache:
                    hidden_states, new_kv = layer_out
                    new_key_values.append(new_kv)
                else:
                    hidden_states = layer_out

        if use_cache:
            return hidden_states, new_key_values
        return hidden_states

    def _forward_with_cache_manager(
        self,
        hidden_states,
        attention_mask,
        rotary_pos_emb,
        kv_cache_manager,
        cache_position,
        position_ids=None,
    ):
        """Forward pass using stateful KVCacheManager.

        This method is used during inference when stateful cache is enabled.
        Each layer updates and reads from the shared cache manager.

        Args:
            hidden_states: [b, s, h]
            attention_mask: [b, 1, s, s]
            rotary_pos_emb: Rotary position embedding
            kv_cache_manager: KVCacheManager instance
            cache_position: Starting position in cache
            position_ids: Optional position IDs [b, s]

        Returns:
            hidden_states: [b, s, h]
        """
        for i, layer in enumerate(self.layers):
            hidden_states = layer.forward_with_cache(
                hidden_states,
                attention_mask,
                rotary_pos_emb,
                kv_cache_manager,
                layer_idx=i,
                cache_position=cache_position,
                position_ids=position_ids,
            )
        return hidden_states
