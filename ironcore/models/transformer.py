# Copyright (c) 2025-2026 Jaegeun Han
#
# SPDX-License-Identifier: Apache-2.0

from contextlib import ExitStack, contextmanager
from functools import partial
from typing import Any

import torch
from torch import nn

from ironcore.config import MainConfig
from ironcore.config.recompute import recompute_segments
from ironcore.layers import BaseModule
from ironcore.layers.attention import Attention
from ironcore.layers.layernorm import get_norm
from ironcore.layers.mlp import MLP
from ironcore.parallel import parallel_states
from ironcore.parallel.random import checkpoint_with_tensor_parallel_rng, tensor_parallel_rng_fork
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

    def _attention_subblock(self, hidden_states, attention_mask, rotary_pos_emb, position_ids):
        """Attention sub-block: layernorm + attention + output proj + residual."""
        batch_size = hidden_states.size(0)
        seq_len = hidden_states.size(1)

        norm_output = self.input_layernorm(hidden_states)
        query = self.linear_q(norm_output)
        key_value = self.linear_kv(norm_output)
        key, value = torch.chunk(key_value, 2, dim=-1)

        query = query.view(batch_size, seq_len, self.num_local_attention_heads, self.head_dimension)
        key = key.view(batch_size, seq_len, self.num_local_attention_groups, self.head_dimension)
        value = value.view(
            batch_size, seq_len, self.num_local_attention_groups, self.head_dimension
        )

        if rotary_pos_emb:
            query = rotary_pos_emb.forward(query, position_ids)
            key = rotary_pos_emb.forward(key, position_ids)

        attn_output = self.self_attention(
            query, key, value, attention_mask, use_cache=False, past_kv=None
        )
        attention_output = self.attn_output(attn_output)

        if self.config.model.dropout_attn > 0.0:
            with tensor_parallel_rng_fork(self.config.init.seed, attention_output.device):
                attention_output = self.residual_dropout(attention_output)

        if self.config.model.is_granitemoe:
            attention_output = attention_output * self.config.model.granitemoe.residual_multiplier
        return hidden_states + attention_output

    def _mlp_subblock(self, norm_input, moe_token_mask=None):
        """MLP sub-block: layernorm + MLP + residual."""
        norm_output = self.post_attn_layernorm(norm_input)
        mlp_output = (
            self.mlp(norm_output, token_mask=moe_token_mask)
            if self.config.model.moe.use_moe
            else self.mlp(norm_output)
        )
        if self.config.model.is_granitemoe:
            mlp_output = mlp_output * self.config.model.granitemoe.residual_multiplier
        return norm_input + mlp_output

    def _full_layer_subblock(
        self,
        hidden_states,
        attention_mask,
        rotary_pos_emb,
        position_ids=None,
        moe_token_mask=None,
    ):
        """Full layer block: attention + MLP, for full_layer spill granularity."""
        norm_input = self._attention_subblock(
            hidden_states, attention_mask, rotary_pos_emb, position_ids
        )
        return self._mlp_subblock(norm_input, moe_token_mask)

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
        block_kv_cache_manager=None,
        seq_id=None,
        moe_token_mask=None,
    ):
        # Activation spilling path uses custom autograd.Function that
        # spills inputs to host during forward and restores during backward,
        # freeing GPU memory for intermediate activations.
        scheduler = getattr(self, "_offload_scheduler", None)
        spill_active = (
            scheduler is not None and scheduler.spill_manager is not None and self.training
        )

        if spill_active:
            from ironcore.offload.hooks import _SpillCheckpointFn

            # _SpillCheckpointFn.forward() runs the sub-block under torch.no_grad(),
            # so parameters don't contribute to the autograd graph. For apply() to
            # create a grad_fn, at least one tensor input must require_grad.
            # Detach + require_grad ensures the autograd chain stays connected
            # without leaking the original tensor into the saved context.
            if not hidden_states.requires_grad:
                hidden_states = hidden_states.detach().requires_grad_(True)

            if getattr(scheduler, "_activation_spill_granularity", "sub_layer") == "full_layer":
                # Full layer: single spill wrapping attention + MLP
                return _SpillCheckpointFn.apply(
                    self._full_layer_subblock,
                    scheduler,
                    self.layer_idx,
                    0,
                    hidden_states,
                    attention_mask,
                    rotary_pos_emb,
                    position_ids,
                )
            else:
                # Sub-layer: two spills at attention/MLP boundaries
                # Sub-block 1: Attention (spills hidden_states as sub_layer=0)
                norm_input = _SpillCheckpointFn.apply(
                    self._attention_subblock,
                    scheduler,
                    self.layer_idx,
                    0,
                    hidden_states,
                    attention_mask,
                    rotary_pos_emb,
                    position_ids,
                )
                # Sub-block 2: MLP (spills norm_input as sub_layer=1)
                output = _SpillCheckpointFn.apply(
                    self._mlp_subblock,
                    scheduler,
                    self.layer_idx,
                    1,
                    norm_input,
                )
                return output

        # Standard forward path (no spilling)
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
        # 1. Explicit cache (use_cache=True / past_key_value)
        # 2. Stateful KVCacheManager
        # 3. Block-based paged KV cache
        # 4. No caching
        if use_cache or past_key_value is not None:
            attn_output = self.self_attention(
                query, key, value, attention_mask, use_cache=use_cache, past_kv=past_key_value
            )
            if use_cache:
                attn_output, new_kv = attn_output
            else:
                new_kv = None
        elif (
            block_kv_cache_manager is not None
            and seq_id is not None
            and block_kv_cache_manager.is_initialized
        ):
            # Paged KV cache path (supports single int or batched list)
            is_batched = isinstance(seq_id, list)
            if is_batched:
                if seq_len > 1:
                    for i, sid in enumerate(seq_id):
                        block_kv_cache_manager.write_prefill(
                            self.layer_idx, sid, key[i : i + 1], value[i : i + 1]
                        )
                else:
                    block_kv_cache_manager.write_decode_batched(self.layer_idx, seq_id, key, value)
                full_key, full_value = block_kv_cache_manager.get_layer_kv_gathered_batched(
                    self.layer_idx, seq_id
                )
            else:
                if seq_len > 1:
                    block_kv_cache_manager.write_prefill(self.layer_idx, seq_id, key, value)
                else:
                    block_kv_cache_manager.write_decode(self.layer_idx, seq_id, key, value)
                full_key, full_value = block_kv_cache_manager.get_layer_kv_gathered(
                    self.layer_idx, seq_id
                )
            attn_output = self.self_attention(query, full_key, full_value, attention_mask)
            new_kv = None
        elif (
            kv_cache_manager is not None
            and cache_position is not None
            and kv_cache_manager.is_initialized
        ):
            # Legacy KV cache manager path
            full_key, full_value = kv_cache_manager.update_layer(
                self.layer_idx, key, value, position=cache_position
            )
            attn_output = self.self_attention(query, full_key, full_value, attention_mask)
            new_kv = None
        else:
            attn_output = self.self_attention(
                query, key, value, attention_mask, use_cache=False, past_kv=None
            )
            new_kv = None

        # output projection
        attention_output = self.attn_output(attn_output)

        if self.config.model.dropout_attn > 0.0:
            with tensor_parallel_rng_fork(self.config.init.seed, attention_output.device):
                attention_output = self.residual_dropout(attention_output)

        residual = hidden_states
        if self.config.model.is_granitemoe:
            attention_output = attention_output * self.config.model.granitemoe.residual_multiplier
        norm_input = residual + attention_output
        norm_output = self.post_attn_layernorm(norm_input)
        mlp_output = (
            self.mlp(norm_output, token_mask=moe_token_mask)
            if self.config.model.moe.use_moe
            else self.mlp(norm_output)
        )
        if self.config.model.is_granitemoe:
            mlp_output = mlp_output * self.config.model.granitemoe.residual_multiplier
        output = norm_input + mlp_output

        if use_cache or kv_cache_manager is not None or block_kv_cache_manager is not None:
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
        block_kv_cache_manager=None,
        seq_id=None,
        moe_token_mask=None,
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
            block_kv_cache_manager=block_kv_cache_manager,
            seq_id=seq_id,
            moe_token_mask=moe_token_mask,
        )

    def _checkpoint_forward_with_aux(self, *args: Any) -> tuple[torch.Tensor, torch.Tensor]:
        """Expose auxiliary loss as a checkpoint output, including reentrant mode."""
        output = self.custom_forward(*args)
        return output, self.mlp.get_aux_loss()


class TransformerModel(BaseModule):
    def __init__(self, config: MainConfig):
        super().__init__(config)
        self.layers = nn.ModuleList()
        for i in range(config.model.num_layers):
            if config.model.is_gemma4:
                from ironcore.layers.gemma4 import Gemma4Layer

                layer = Gemma4Layer(config, i)
            else:
                layer = TransformerLayer(config)
            layer.layer_idx = i
            self.layers.append(layer)

        if config.model.is_gemma4 and config.model.gemma4.hidden_size_per_layer_input:
            from ironcore.layers.gemma4 import Gemma4PerLayerEmbedding, Gemma4RMSNorm

            gemma = config.model.gemma4
            self.embed_tokens_per_layer = Gemma4PerLayerEmbedding(config)
            self.per_layer_model_projection = ColumnParallelLinear(
                config,
                config.model.d_model,
                config.model.num_layers * gemma.hidden_size_per_layer_input,
                bias=False,
                gather_output=True,
            )
            self.per_layer_projection_norm = Gemma4RMSNorm(
                gemma.hidden_size_per_layer_input, config.model.ln_eps
            )

        # Activation checkpointing configuration
        self.recompute_interval = config.operation.recompute_interval
        self.recompute_skip_count = config.operation.recompute_skip_count
        self.activation_recompute = (
            config.operation.activation_recompute and self.recompute_interval > 0
        )
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
                return getattr(self.config.parallel, "use_fsdp", False)
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
        block_kv_cache_manager=None,
        seq_id=None,
        input_ids=None,
        moe_token_mask=None,
    ):
        if self.config.model.is_gemma4:
            if kv_cache_manager is not None or block_kv_cache_manager is not None:
                raise ValueError("Gemma 4 requires explicit KV tuples, not external cache managers")
            return self._forward_gemma4(
                hidden_states,
                attention_mask,
                position_ids,
                input_ids,
                use_cache,
                past_key_values,
            )
        new_key_values = [] if use_cache else None

        is_fsdp = self._is_fsdp_enabled()
        scheduler = getattr(self, "_offload_scheduler", None)

        # Spilled inputs use their own replay engine rather than ordinary GPU
        # checkpointing. Group planning below controls the stored input boundaries.
        activation_spill_active = scheduler is not None and scheduler.spill_manager is not None
        use_layer_checkpointing = (
            self.activation_recompute
            and self.training
            and not use_cache
            and kv_cache_manager is None
            and block_kv_cache_manager is None
            and not is_fsdp
            and not activation_spill_active
        )
        checkpoint_reentrant = self.use_reentrant
        if (
            self.config.model.moe.use_moe
            and self.config.model.moe.expert_model_parallel_size == 1
            and parallel_states.get_data_parallel_world_size() > 1
        ):
            # Dynamic idle experts require DDP's unused-parameter traversal.
            # Reentrant checkpointing hides this graph and can mark parameters
            # ready twice; preserve recomputation with the non-reentrant engine.
            checkpoint_reentrant = False

        # Weight streaming: model stays on CPU but scheduler loads weights to GPU
        # per-layer. Move hidden_states (and related tensors) to GPU for computation.
        original_device = hidden_states.device
        weight_streaming = scheduler is not None and scheduler.weight_streaming_enabled
        if weight_streaming:
            compute_device = scheduler.device
            hidden_states = hidden_states.to(compute_device)
            if attention_mask is not None:
                attention_mask = attention_mask.to(compute_device)
            if rotary_pos_emb is not None and isinstance(rotary_pos_emb, torch.Tensor):
                rotary_pos_emb = rotary_pos_emb.to(compute_device)
            if position_ids is not None and isinstance(position_ids, torch.Tensor):
                position_ids = position_ids.to(compute_device)

        if (
            (self.grouped_recompute or (weight_streaming and not activation_spill_active))
            and self.training
            and not use_cache
            and kv_cache_manager is None
            and block_kv_cache_manager is None
            and (activation_spill_active or use_layer_checkpointing)
        ):
            hidden_states = self._forward_grouped(
                hidden_states,
                attention_mask,
                rotary_pos_emb,
                position_ids,
                moe_token_mask,
                scheduler if activation_spill_active or weight_streaming else None,
                checkpoint_reentrant,
            )
            if weight_streaming and hidden_states.device != original_device:
                hidden_states = hidden_states.to(original_device)
            return hidden_states

        for i, layer in enumerate(self.layers):
            past_kv = past_key_values[i] if past_key_values is not None else None

            # Weight streaming: ensure layer weights are on GPU before execution
            if scheduler is not None:
                scheduler.on_layer_start(i)

            if use_layer_checkpointing:
                checkpoint_kwargs = {}
                if self.config.model.moe.use_moe:
                    checkpoint_kwargs["recompute_context"] = layer.mlp.recompute_context
                layer_out = checkpoint_with_tensor_parallel_rng(
                    layer._checkpoint_forward_with_aux
                    if self.config.model.moe.use_moe
                    else layer.custom_forward,
                    hidden_states,
                    attention_mask,
                    rotary_pos_emb,
                    position_ids,
                    False,  # use_cache
                    None,  # past_key_value
                    None,  # kv_cache_manager
                    None,  # cache_position
                    None,  # block_kv_cache_manager
                    None,  # seq_id
                    moe_token_mask,
                    use_reentrant=checkpoint_reentrant,
                    **checkpoint_kwargs,
                )
                if self.config.model.moe.use_moe:
                    layer_out, aux_loss = layer_out
                    layer.mlp._aux_loss = aux_loss
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
                    block_kv_cache_manager=block_kv_cache_manager,
                    seq_id=seq_id,
                    moe_token_mask=moe_token_mask,
                )

            # Weight streaming: mark layer as done (weights stay for backward)
            if scheduler is not None:
                scheduler.on_layer_end(i)

            if use_cache or kv_cache_manager is not None or block_kv_cache_manager is not None:
                hidden_states, new_kv = layer_out
                if use_cache:
                    new_key_values.append(new_kv)
            else:
                hidden_states = layer_out

        # Weight streaming: restore hidden_states to original device for downstream
        # layers (output_layernorm, output head) which stay on CPU.
        if weight_streaming and hidden_states.device != original_device:
            hidden_states = hidden_states.to(original_device)

        if use_cache or kv_cache_manager is not None or block_kv_cache_manager is not None:
            return hidden_states, new_key_values
        return hidden_states

    @contextmanager
    def _group_recompute_context(self, start, end):
        with ExitStack() as stack:
            if self.config.model.moe.use_moe:
                for layer in self.layers[start:end]:
                    stack.enter_context(layer.mlp.recompute_context())
            yield

    def get_recompute_segments(self) -> tuple[tuple[int, int], ...]:
        return recompute_segments(
            len(self.layers), self.recompute_interval, self.recompute_skip_count
        )

    @property
    def grouped_recompute(self) -> bool:
        return self.recompute_interval > 1 or (self.recompute_skip_count or 0) > 0

    def _forward_grouped(self, hidden, mask, rotary, positions, token_mask, scheduler, reentrant):
        """Recompute segments, independent of the activation storage device."""
        for start, end in self.get_recompute_segments():
            if scheduler is not None:
                from ironcore.offload.recompute import recompute_streamed_segment

                def streamed(layer, value, attention_mask, rotary_pos_emb, position_ids, moe_mask):
                    output = layer._full_layer_subblock(
                        value, attention_mask, rotary_pos_emb, position_ids, moe_mask
                    )
                    if self.config.model.moe.use_moe:
                        return output, layer.mlp.get_aux_loss()
                    return output

                functions = [partial(streamed, layer) for layer in self.layers[start:end]]
                if self.config.model.moe.use_moe:
                    for function, layer in zip(functions, self.layers[start:end], strict=True):
                        function.recompute_context = layer.mlp.recompute_context
                output = recompute_streamed_segment(
                    functions,
                    scheduler,
                    start,
                    hidden,
                    mask,
                    rotary,
                    positions,
                    token_mask,
                    offload_input=scheduler.spill_manager is not None,
                )
            else:
                output = checkpoint_with_tensor_parallel_rng(
                    partial(self._run_group, start, end),
                    hidden,
                    mask,
                    rotary,
                    positions,
                    token_mask,
                    use_reentrant=reentrant,
                    recompute_context=partial(self._group_recompute_context, start, end),
                )
            if self.config.model.moe.use_moe:
                hidden, *losses = output
                for layer, loss in zip(self.layers[start:end], losses, strict=True):
                    layer.mlp._aux_loss = loss
            else:
                hidden = output
        return hidden

    def _run_group(self, start, end, hidden, mask, rotary, positions, token_mask):
        losses = []
        for layer in self.layers[start:end]:
            hidden = layer.custom_forward(
                hidden, mask, rotary, position_ids=positions, moe_token_mask=token_mask
            )
            if self.config.model.moe.use_moe:
                losses.append(layer.mlp.get_aux_loss())
        return (hidden, *losses) if losses else hidden

    def _forward_gemma4(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None,
        position_ids: torch.Tensor,
        input_ids: torch.Tensor,
        use_cache: bool,
        past_key_values: list | None,
    ) -> torch.Tensor | tuple[torch.Tensor, list]:
        """Run Gemma layers with explicit KV tensors, safe for recomputation."""
        gemma = self.config.model.gemma4
        per_layer_inputs = None
        if gemma.hidden_size_per_layer_input:
            if input_ids is None:
                raise ValueError("Gemma 4 PLE requires input_ids alongside token embeddings")
            shape = (
                *input_ids.shape,
                self.config.model.num_layers,
                gemma.hidden_size_per_layer_input,
            )
            token_scale = torch.tensor(
                gemma.hidden_size_per_layer_input**0.5,
                device=hidden_states.device,
                dtype=hidden_states.dtype,
            )
            token_inputs = (self.embed_tokens_per_layer(input_ids) * token_scale).reshape(shape)
            projected = (
                self.per_layer_model_projection(hidden_states) * self.config.model.d_model**-0.5
            )
            projected = self.per_layer_projection_norm(projected.reshape(shape))
            per_layer_inputs = (token_inputs + projected) * 2.0**-0.5
        shared = {}
        cache = []
        scheduler = getattr(self, "_offload_scheduler", None)
        spill_active = scheduler is not None and scheduler.spill_manager is not None
        streamed_gpu = (
            scheduler is not None and scheduler.weight_streaming_enabled and not spill_active
        )
        if (
            (self.grouped_recompute or streamed_gpu)
            and self.training
            and not use_cache
            and (spill_active or (self.activation_recompute and not self._is_fsdp_enabled()))
        ):
            return self._forward_gemma4_grouped(
                hidden_states,
                attention_mask,
                position_ids,
                per_layer_inputs,
                scheduler if spill_active or streamed_gpu else None,
            )
        for i, layer in enumerate(self.layers):
            if scheduler is not None:
                scheduler.on_layer_start(i)
            kind = gemma.layer_types[i]
            past = past_key_values[i] if past_key_values is not None else None
            kv = shared.get(kind) if layer.self_attn.is_shared else None
            per_layer_input = per_layer_inputs[:, :, i] if per_layer_inputs is not None else None
            args = hidden_states, attention_mask, position_ids, past, kv, per_layer_input
            if scheduler is not None and scheduler.spill_manager is not None and self.training:
                from ironcore.offload.hooks import _SpillCheckpointFn

                hidden_states = _SpillCheckpointFn.apply(
                    layer.forward_spilled,
                    scheduler,
                    i,
                    0,
                    hidden_states,
                    attention_mask,
                    position_ids,
                )
                new_kv = None
            elif (
                self.activation_recompute
                and self.training
                and not use_cache
                and not self._is_fsdp_enabled()
            ):
                key, value = kv if kv is not None else (None, None)
                hidden_states, new_key, new_value = checkpoint_with_tensor_parallel_rng(
                    layer.forward_checkpointed,
                    hidden_states,
                    attention_mask,
                    position_ids,
                    key,
                    value,
                    per_layer_input,
                    use_reentrant=self.use_reentrant,
                )
                new_kv = new_key, new_value
            else:
                hidden_states, new_kv = layer(*args)
            if gemma.num_kv_shared_layers and not layer.self_attn.is_shared:
                shared[kind] = new_kv
            if use_cache:
                cache.append(new_kv)
            if scheduler is not None:
                scheduler.on_layer_end(i)
        return (hidden_states, cache) if use_cache else hidden_states

    def _forward_gemma4_grouped(self, hidden, mask, positions, per_layer_inputs, scheduler):
        shared = {}
        for start, end in self.get_recompute_segments():
            if scheduler is not None:
                from ironcore.offload.recompute import recompute_streamed_segment

                hidden = recompute_streamed_segment(
                    [layer.forward_spilled for layer in self.layers[start:end]],
                    scheduler,
                    start,
                    hidden,
                    mask,
                    positions,
                    offload_input=scheduler.spill_manager is not None,
                )
            else:
                kinds = tuple(shared)
                hidden, *kv_outputs = checkpoint_with_tensor_parallel_rng(
                    partial(self._run_gemma4_group, start, end, kinds),
                    hidden,
                    mask,
                    positions,
                    per_layer_inputs,
                    *(tensor for kind in kinds for tensor in shared[kind]),
                    use_reentrant=self.use_reentrant,
                )
                output_kinds = tuple(
                    dict.fromkeys(
                        (
                            *kinds,
                            *(
                                self.config.model.gemma4.layer_types[i]
                                for i in range(start, end)
                                if self.config.model.gemma4.num_kv_shared_layers
                                and not self.layers[i].self_attn.is_shared
                            ),
                        )
                    )
                )
                shared = {
                    kind: tuple(kv_outputs[2 * i : 2 * i + 2])
                    for i, kind in enumerate(output_kinds)
                }
        return hidden

    def _run_gemma4_group(self, start, end, kinds, hidden, mask, positions, ple, *kv_inputs):
        shared = {kind: tuple(kv_inputs[2 * i : 2 * i + 2]) for i, kind in enumerate(kinds)}
        gemma = self.config.model.gemma4
        for i in range(start, end):
            layer = self.layers[i]
            kind = gemma.layer_types[i]
            hidden, new_kv = layer(
                hidden,
                mask,
                positions,
                None,
                shared.get(kind) if layer.self_attn.is_shared else None,
                ple[:, :, i] if ple is not None else None,
            )
            if gemma.num_kv_shared_layers and not layer.self_attn.is_shared:
                shared[kind] = new_kv
        return hidden, *(tensor for kv in shared.values() for tensor in kv)
