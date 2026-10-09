# Copyright (c) 2025-2026 Jaegeun Han
#
# SPDX-License-Identifier: Apache-2.0
# configure language model sequential

import zlib

import torch
import torch.distributed as dist
import torch.nn.functional as F

from ironcore import get_tokenizer
from ironcore.config import MainConfig
from ironcore.layers import BaseModule, LanguageModelEmbedding
from ironcore.layers.layernorm import get_norm
from ironcore.layers.positional_embedding import RotaryPositionalEmbedding
from ironcore.models import get_model_provider_func
from ironcore.parallel import parallel_states
from ironcore.parallel.tensor_parallel import (
    ColumnParallelLinear,
    vocab_parallel_cross_entropy,
)
from ironcore.parallel.tensor_parallel.comm import gather_from_model_parallel_workers


class LanguageModel(BaseModule):
    def __init__(
        self,
        config: MainConfig,
        loss_fn: torch.nn.modules.loss._Loss = F.cross_entropy,
    ):
        super().__init__(config)

        from ironcore.config.config_blockwise import validate_blockwise_mlp
        from ironcore.config.config_context_parallel import validate_context_parallel
        from ironcore.config.config_gemma4 import validate_gemma4_runtime

        validate_gemma4_runtime(config)
        validate_blockwise_mlp(config)
        validate_context_parallel(config)
        if (
            config.trainer.context_parallel_size
            != parallel_states.get_context_parallel_world_size()
        ):
            raise ValueError(
                "Initialize the configured context parallel group before constructing the model"
            )

        tokenizer = get_tokenizer()

        self.eod_mask_loss = config.model.eod_mask_loss
        self.reset_position_ids = config.model.reset_position_ids
        self.reset_attention_mask = config.model.reset_attention_mask
        self.fp16_lm_cross_entropy = config.model.fp16_lm_cross_entropy

        # model components initialization
        self.embedding = LanguageModelEmbedding(config)
        self.rotary_pos_emb = None
        if config.model.positional_embedding.type == "rope" and not config.model.is_gemma4:
            self.rotary_pos_emb = RotaryPositionalEmbedding(
                config.model.d_model // config.model.num_attention_heads,
                config.model.max_position_embeddings,
                base=config.model.positional_embedding.base,
                scale=config.model.positional_embedding.scaling_factor,
                offset=config.model.positional_embedding.offset,
            )

        model_provider_func = get_model_provider_func(config)
        self.model = model_provider_func(config)
        if config.model.is_gemma4:
            from ironcore.layers.gemma4 import Gemma4RMSNorm

            self.output_layernorm = Gemma4RMSNorm(config.model.d_model, config.model.ln_eps)
        else:
            self.output_layernorm = get_norm(config)

        if config.model.untie_embed:
            self.output_layer = ColumnParallelLinear(
                config, config.model.d_model, tokenizer.padded_vocab_size, bias=False
            )

        self.loss_fn = loss_fn
        self.padding_start_idx = tokenizer.vocab_size

        # Initialize KV cache manager for inference
        self.kv_cache_manager = None
        if (
            config.model.kv_cache.enabled
            and config.trainer.context_parallel_size == 1
            and not config.model.kv_cache.use_paged
            and not config.model.is_gemma4
        ):
            from ironcore.layers.kv_cache import KVCacheManager

            self.kv_cache_manager = KVCacheManager(config)

        # Initialize block-based paged KV cache (alternative to kv_cache_manager)
        self.block_kv_cache_manager = None
        if config.model.kv_cache.enabled and config.model.kv_cache.use_paged:
            from ironcore.layers.block_kv_cache import BlockKVCacheManager

            self.block_kv_cache_manager = BlockKVCacheManager(config)

        self.init_weights()
        if config.peft.method == "lora":
            from ironcore.peft.lora import LoRALinear

            # BaseModule initializes all parameters; restore zero-output adapters.
            for name, module in self.named_modules():
                if isinstance(module, LoRALinear):
                    generator = torch.Generator(device=module.lora_A.device)
                    generator.manual_seed((config.init.seed + zlib.crc32(name.encode())) % (2**63))
                    module._init_weights(generator)
        if config.model.is_gemma4 and config.model.gemma4.hidden_size_per_layer_input:
            ple = self.model.embed_tokens_per_layer
            with torch.no_grad():
                if ple.local_padding_idx is not None:
                    ple.weight[ple.local_padding_idx].zero_()

        # Initialize VocabParallelEmbedding (zeros padding, registers hooks)
        if hasattr(self.embedding.word_embeddings, "init_weight"):
            self.embedding.word_embeddings.init_weight()

    def forward(
        self,
        input_ids,
        labels=None,
        position_ids=None,
        use_cache=False,
        past_key_values=None,
        cache_position=None,
        block_kv_cache_manager=None,
        seq_id=None,
        attention_mask=None,
        loss_sample_ids=None,
    ):
        """
        Forward pass through language model.
        """
        input_ids = input_ids.to(self.device, non_blocking=True)
        if labels is not None:
            labels = labels.to(self.device, non_blocking=True)

        cp_active = self.config.trainer.context_parallel_size > 1
        if cp_active and (
            use_cache
            or past_key_values is not None
            or block_kv_cache_manager is not None
            or seq_id is not None
            or attention_mask is not None
            or loss_sample_ids is not None
            or (
                cache_position is not None
                and (isinstance(cache_position, torch.Tensor) or cache_position != 0)
            )
        ):
            raise ValueError(
                "Context parallel currently supports causal full-sequence scoring without masks/caches"
            )

        bkv = block_kv_cache_manager
        if bkv is None and self.block_kv_cache_manager is not None and not self.training:
            bkv = self.block_kv_cache_manager

        # Determine cache position
        # For batched paged decode, use per-sequence positions from block cache
        if cache_position is None:
            if bkv is not None and seq_id is not None:
                if isinstance(seq_id, list):
                    seq_id_t = torch.tensor(seq_id, dtype=torch.long, device=input_ids.device)
                    cache_position = bkv.token_positions[seq_id_t]
                else:
                    cache_position = int(bkv.token_positions[seq_id].item())
            else:
                cache_position = 0
            if use_cache and past_key_values is not None and len(past_key_values) > 0:
                first_layer_kv = past_key_values[0]
                if (
                    isinstance(first_layer_kv, tuple | list)
                    and len(first_layer_kv) >= 2
                    and first_layer_kv[0] is not None
                ):
                    past_key = first_layer_kv[0]
                    cache_position = past_key.size(1)

        computed_attention_mask, computed_position_ids, loss_mask = self.get_masks_and_position_ids(
            input_ids, labels, cache_position=cache_position, use_cache=use_cache
        )
        if attention_mask is None:
            attention_mask = computed_attention_mask
        else:
            attention_mask = attention_mask.to(device=input_ids.device, dtype=torch.bool)
            if attention_mask.dim() == 3:
                attention_mask = attention_mask.unsqueeze(1)
            # Packed SFT collators describe document blocks. Each block must
            # also remain causal, otherwise responses leak into their prompts.
            if not use_cache and attention_mask.size(-1) == input_ids.size(1):
                causal = torch.ones(
                    input_ids.size(1), input_ids.size(1), device=input_ids.device, dtype=torch.bool
                ).tril()
                attention_mask = attention_mask & causal
        if position_ids is None:
            position_ids = computed_position_ids
        else:
            position_ids = position_ids.to(self.device, non_blocking=True)

        original_sequence_length = input_ids.size(1)
        if cp_active:
            from ironcore.parallel.context_parallel import partition_context_inputs

            input_ids, labels, position_ids, original_sequence_length = partition_context_inputs(
                input_ids,
                labels,
                position_ids,
            )
            loss_mask = (
                (labels != -100).float()
                if labels is not None
                else torch.ones_like(input_ids, dtype=torch.float)
            )

        x = self.embedding(input_ids, position_ids)
        if (
            self.training
            and self.config.peft.method == "lora"
            and (
                self.config.offload.activation_spill
                or (
                    self.config.operation.activation_recompute
                    and self.config.operation.recompute_strategy == "optimized"
                )
            )
            and not x.requires_grad
        ):
            # Reentrant checkpointing needs a differentiable input even when
            # the embedding is frozen; adapter parameters remain trainable.
            x.requires_grad_(True)

        model_kwargs = {"input_ids": input_ids} if self.config.model.is_gemma4 else {}
        if cp_active and self.config.model.moe.use_moe:
            from ironcore.parallel import parallel_states as ps

            offset = ps.get_context_parallel_rank() * input_ids.size(1)
            valid = torch.arange(input_ids.size(1), device=input_ids.device) + offset
            model_kwargs["moe_token_mask"] = (valid < original_sequence_length)[None].expand_as(
                input_ids
            )
        model_out = self.model(
            x,
            attention_mask,
            self.rotary_pos_emb,
            position_ids=position_ids,
            use_cache=use_cache,
            past_key_values=past_key_values,
            kv_cache_manager=self.kv_cache_manager if not self.training else None,
            cache_position=cache_position if not self.training else None,
            block_kv_cache_manager=bkv,
            seq_id=seq_id,
            **model_kwargs,
        )

        has_cache = (
            use_cache
            or (self.kv_cache_manager is not None and not self.training)
            or (bkv is not None)
        )
        if has_cache:
            lm_output, new_key_values = model_out
        else:
            lm_output = model_out
            new_key_values = None

        lm_output = self.output_layernorm(lm_output)

        if labels is not None and self.config.trainer.recompute_linear_ce:
            from ironcore.layers.linear_cross_entropy import linear_cross_entropy
            from ironcore.parallel.tensor_parallel import comm

            if self.config.model.untie_embed and self.output_layer.bias is not None:
                raise ValueError("Recomputed linear CE currently requires a bias-free output head")
            hidden = comm.copy_inputs_to_model_parallel_workers(lm_output)
            weight = (
                self.output_layer.weight
                if self.config.model.untie_embed
                else self.embedding.word_embeddings.weight
            )
            per_token = linear_cross_entropy(
                hidden,
                weight,
                labels,
                self.config.trainer.loss_chunk_size or 1024,
                self.padding_start_idx,
                transposed=self.config.model.untie_embed,
                softcap=self.config.model.gemma4.final_logit_softcapping
                if self.config.model.is_gemma4
                else None,
            )
            if loss_sample_ids is not None:
                return self.loss_fn(per_token, loss_mask, sample_ids=loss_sample_ids)
            if cp_active:
                from ironcore.parallel.context_parallel import (
                    context_parallel_sample_mean,
                    context_parallel_token_mean,
                )

                reduction = (
                    context_parallel_sample_mean
                    if self.config.data.task_type == "sft"
                    else context_parallel_token_mean
                )
                return reduction(per_token, loss_mask)
            return self.loss_fn(per_token, loss_mask)

        if self.config.model.untie_embed:
            logits_parallel = self.output_layer(lm_output)
        else:
            from ironcore.parallel.tensor_parallel import comm

            input_parallel = comm.copy_inputs_to_model_parallel_workers(lm_output)
            logits_parallel = F.linear(input_parallel, self.embedding.word_embeddings.weight)

        if self.config.model.is_gemma4:
            softcap = self.config.model.gemma4.final_logit_softcapping
            if softcap is not None:
                logits_parallel = (logits_parallel / softcap).tanh() * softcap

        if labels is None:
            logits = gather_from_model_parallel_workers(
                logits_parallel,
                {"column_parallel": True, "concatenated_weights": 1},
            )
            if cp_active:
                from ironcore.parallel.context_parallel import gather_context_parallel

                logits = gather_context_parallel(logits)[:, :original_sequence_length]
            # Always return a (logits, new_key_values) tuple, regardless of
            # whether a KV cache is active. Callers that do not need the cache
            # simply ignore the second element. Returning a bare tensor when
            # has_cache is False breaks callers (DPO trainer, eval loop) that
            # always unpack the cache — see Fable issue #56.
            return logits, new_key_values

        losses = self.compute_loss_from_logits(
            logits_parallel,
            labels,
            loss_mask,
            self.fp16_lm_cross_entropy,
            self.padding_start_idx,
            loss_sample_ids=loss_sample_ids,
        )
        return losses

    def compute_loss_from_logits(
        self,
        logits,
        labels,
        loss_mask,
        fp16_lm_cross_entropy=False,
        padding_start_idx=None,
        loss_sample_ids=None,
    ):
        """Compute loss from logits using vocab_parallel_cross_entropy.

        Args:
            logits: [batch, seq_len, vocab_size] or [batch, seq_len, vocab_size/tp]
            labels: [batch, seq_len] ground truth token IDs
            loss_mask: [batch, seq_len] valid token mask
            fp16_lm_cross_entropy: Whether to use fp16 for cross entropy
            padding_start_idx: Index where padding tokens start in vocab

        Returns:
            Scalar loss value
        """
        labels = labels.contiguous()

        loss_dtype = torch.float16 if fp16_lm_cross_entropy else torch.float32
        chunk_size = self.config.trainer.loss_chunk_size
        if chunk_size is not None:
            if chunk_size <= 0:
                raise ValueError("trainer.loss_chunk_size must be positive or None")
            # Cast only the current token chunk. Casting the complete vocabulary
            # logits before splitting would retain the large FP32 allocation.
            flat_logits = logits.reshape(-1, logits.size(-1))
            flat_labels = labels.reshape(-1)
            chunks = [
                vocab_parallel_cross_entropy(
                    flat_logits[start : start + chunk_size].to(loss_dtype),
                    flat_labels[start : start + chunk_size],
                    padding_start_idx=padding_start_idx,
                )
                for start in range(0, flat_labels.numel(), chunk_size)
            ]
            per_token_losses = torch.cat(chunks).view_as(labels)
        else:
            per_token_losses = vocab_parallel_cross_entropy(
                vocab_parallel_logits=logits.to(loss_dtype),
                labels=labels,
                padding_start_idx=padding_start_idx,
            ).contiguous()

        if self.config.trainer.context_parallel_size > 1:
            from ironcore.parallel.context_parallel import (
                context_parallel_sample_mean,
                context_parallel_token_mean,
            )

            reduction = (
                context_parallel_sample_mean
                if self.config.data.task_type == "sft"
                else context_parallel_token_mean
            )
            return reduction(per_token_losses, loss_mask)
        if loss_sample_ids is not None:
            loss = self.loss_fn(per_token_losses, loss_mask, sample_ids=loss_sample_ids)
        else:
            loss = self.loss_fn(per_token_losses, loss_mask)

        return loss

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 128,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = 0,
        do_sample: bool = False,
        eos_token_id: int | list[int] | None = None,
    ) -> torch.Tensor:
        """
        Autoregressive generation with KV cache.
        Supports legacy KVCacheManager or block-based paged cache.
        """
        if self.config.trainer.context_parallel_size > 1:
            raise ValueError("Context parallel generation and KV caches are not supported")
        batch_size = input_ids.size(0)
        generated = input_ids.clone()
        past_key_values = None
        done = torch.zeros(batch_size, dtype=torch.bool, device=input_ids.device)
        next_token = input_ids
        eos_tokens = (
            torch.as_tensor(eos_token_id, device=input_ids.device).reshape(-1)
            if eos_token_id is not None
            else None
        )

        use_stateful = self.kv_cache_manager is not None
        use_paged = self.block_kv_cache_manager is not None and not use_stateful

        if use_stateful:
            self.initialize_cache(batch_size, input_ids.device)
        elif use_paged:
            self.initialize_cache(batch_size, input_ids.device)
            assert self.block_kv_cache_manager is not None
            prompt_len = input_ids.size(1)
            blocks_needed = (
                prompt_len + self.block_kv_cache_manager.block_size - 1
            ) // self.block_kv_cache_manager.block_size
            self.block_kv_cache_manager.allocate_blocks(0, blocks_needed)

        for step in range(max_new_tokens):
            cur_input = input_ids if step == 0 else next_token

            if use_stateful:
                cur_cache_pos = self.kv_cache_manager.get_cache_position()
                out = self.forward(
                    cur_input,
                    labels=None,
                    use_cache=False,
                    cache_position=cur_cache_pos,
                )
                logits, _ = out
            elif use_paged:
                if batch_size > 1:
                    raise ValueError(
                        "Paged KV cache generate() only supports batch_size=1. "
                        "For batched generation, use generate_rollouts_paged()."
                    )
                out = self.forward(
                    cur_input,
                    labels=None,
                    use_cache=False,
                    seq_id=0,
                )
                logits, _ = out
                # Advance position after all layers have written
                tokens_written = cur_input.size(1)
                self.advance_cache_position(0, tokens_written)
            else:
                out = self.forward(
                    cur_input,
                    labels=None,
                    use_cache=True,
                    past_key_values=past_key_values,
                )
                logits, past_key_values = out

            next_logits = logits[:, -1, :]
            # forward(labels=None) already gathers the full vocabulary on every rank.

            # Restrict sampling to the real vocabulary. The tied output embedding
            # is padded to `padded_vocab_size` for TP alignment; those padding
            # rows are zero-initialised and would otherwise dominate argmax /
            # skew softmax probabilities. Mask them to -inf so sampling can never
            # emit a token id >= vocab_size.
            if next_logits.size(-1) > self.padding_start_idx:
                next_logits = next_logits[..., : self.padding_start_idx]

            next_token = self._sample(next_logits, temperature, top_p, top_k, do_sample)

            if do_sample and parallel_states.get_tensor_model_parallel_world_size() > 1:
                dist.broadcast(
                    next_token,
                    src=0,
                    group=parallel_states.get_tensor_model_parallel_group(),
                )

            if eos_tokens is not None:
                new_done = torch.isin(next_token.squeeze(1), eos_tokens) | done
                if new_done.all():
                    break
                done = new_done

            generated = torch.cat([generated, next_token], dim=1)

        return generated

    def _sample(self, logits, temperature, top_p, top_k, do_sample):
        if not do_sample:
            return logits.argmax(dim=-1, keepdim=True)
        if temperature != 1.0:
            logits = logits / temperature
        if top_k > 0:
            top_k = min(top_k, logits.size(-1))
            kth_vals = logits.topk(top_k, dim=-1).values[:, -1, None]
            logits = logits.masked_fill(logits < kth_vals, float("-inf"))
        if top_p < 1.0:
            sorted_logits, sorted_idx = logits.sort(dim=-1, descending=True)
            cumprobs = sorted_logits.softmax(dim=-1).cumsum(dim=-1)
            remove = (cumprobs - sorted_logits.softmax(dim=-1)) > top_p
            sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))
            logits = torch.full_like(logits, float("-inf")).scatter_(1, sorted_idx, sorted_logits)
        probs = logits.softmax(dim=-1)
        return torch.multinomial(probs, num_samples=1)

    def get_masks_and_position_ids(self, input_ids, labels=None, cache_position=0, use_cache=False):
        att_mask_batch = input_ids.size(0) if input_ids.dim() == 2 else 1
        seq_len = input_ids.size(1)
        if isinstance(cache_position, torch.Tensor):
            max_cache_pos = cache_position.max().item()
            total_len = int(max_cache_pos + seq_len)
            position_ids = cache_position.unsqueeze(1) + torch.arange(
                seq_len, device=input_ids.device
            )
        else:
            total_len = int(cache_position + seq_len)
            position_ids = (
                torch.arange(cache_position, total_len, dtype=torch.long, device=input_ids.device)
                .unsqueeze(0)
                .expand(att_mask_batch, seq_len)
            )

        # Standard training / full prefill: Attention uses is_causal=True, no mask needed.
        # Inference decode (use_cache or cache_position > 0): explicit mask required.
        if not use_cache and not isinstance(cache_position, torch.Tensor) and cache_position == 0:
            attention_mask = None
        elif seq_len == 1 and not isinstance(cache_position, torch.Tensor) and cache_position > 0:
            # Single-token decode: attend to all cached positions.
            attention_mask = torch.ones(
                (att_mask_batch, 1, 1, total_len), dtype=torch.bool, device=input_ids.device
            )
        else:
            full_causal_mask = torch.tril(
                torch.ones((total_len, total_len), device=input_ids.device, dtype=torch.bool)
            )
            if not isinstance(cache_position, torch.Tensor):
                if cache_position == 0:
                    # use_cache=True prefill: need explicit mask (Attention won't set is_causal).
                    attention_mask = (
                        full_causal_mask.unsqueeze(0)
                        .unsqueeze(0)
                        .expand(att_mask_batch, 1, total_len, total_len)
                    )
                else:
                    attention_mask = (
                        full_causal_mask[cache_position:total_len, :total_len]
                        .unsqueeze(0)
                        .unsqueeze(0)
                        .expand(att_mask_batch, 1, seq_len, total_len)
                    )
            else:
                q_pos = position_ids.unsqueeze(-1)
                kv_pos = torch.arange(total_len, device=input_ids.device).view(1, 1, -1)
                attention_mask = (q_pos >= kv_pos).unsqueeze(1)

        loss_mask = (
            (labels != -100).float()
            if labels is not None
            else torch.ones(input_ids.size(), dtype=torch.float, device=input_ids.device)
        )
        return attention_mask, position_ids, loss_mask

    def initialize_cache(
        self, batch_size: int, device: torch.device, dtype: torch.dtype | None = None
    ):
        if self.config.trainer.context_parallel_size > 1:
            raise ValueError("Context parallel generation and KV caches are not supported")
        if self.kv_cache_manager is not None:
            self.kv_cache_manager.initialize(batch_size, len(self.model.layers), device, dtype)
        if self.block_kv_cache_manager is not None:
            self.block_kv_cache_manager.initialize(
                batch_size, len(self.model.layers), device, dtype
            )

    def reset_cache(self, batch_indices: list[int] | None = None):
        if self.kv_cache_manager is not None:
            self.kv_cache_manager.reset(batch_indices)
        if self.block_kv_cache_manager is not None:
            bkv = self.block_kv_cache_manager
            if batch_indices is None:
                for sid in range(bkv.block_tables.shape[0]):
                    bkv.free_sequence(sid)
            else:
                for sid in batch_indices:
                    bkv.free_sequence(sid)

    def get_cache_statistics(self) -> dict:
        if self.kv_cache_manager is not None:
            return self.kv_cache_manager.get_statistics()
        if self.block_kv_cache_manager is not None:
            return self.block_kv_cache_manager.get_statistics()
        return {"initialized": False}

    def share_prefix_cache(self, src_seq_id: int, dst_seq_ids: list[int]):
        """Share prefix KV blocks from source to destinations (block cache only)."""
        if self.block_kv_cache_manager is not None:
            self.block_kv_cache_manager.share_prefix(src_seq_id, dst_seq_ids)

    def free_sequence_cache(self, seq_id: int):
        """Free all blocks for a sequence (block cache only)."""
        if self.block_kv_cache_manager is not None:
            self.block_kv_cache_manager.free_sequence(seq_id)

    def advance_cache_position(self, seq_id: int | list[int], tokens: int):
        """Advance token position for sequence(s) (block cache only)."""
        if self.block_kv_cache_manager is not None:
            if isinstance(seq_id, list):
                self.block_kv_cache_manager.advance_positions_batched(seq_id, tokens)
            else:
                self.block_kv_cache_manager.advance_position(seq_id, tokens)
