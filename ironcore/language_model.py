# Copyright (c) 2025-2026 Jaegeun Han
#
# SPDX-License-Identifier: Apache-2.0
# configure language model sequential

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
)
from ironcore.parallel.tensor_parallel.comm import _gather_tensor_along_last_dim


class LanguageModel(BaseModule):
    def __init__(
        self,
        config: MainConfig,
        loss_fn: torch.nn.modules.loss._Loss = F.cross_entropy,
    ):
        super().__init__(config)

        tokenizer = get_tokenizer()

        self.eod_mask_loss = config.model.eod_mask_loss
        self.reset_position_ids = config.model.reset_position_ids
        self.reset_attention_mask = config.model.reset_attention_mask
        self.fp16_lm_cross_entropy = config.model.fp16_lm_cross_entropy

        # model components initialization
        self.embedding = LanguageModelEmbedding(config)
        self.rotary_pos_emb = None
        if config.model.positional_embedding.type == "rope":
            self.rotary_pos_emb = RotaryPositionalEmbedding(
                config.model.d_model // config.model.num_attention_heads,
                config.model.max_position_embeddings,
                base=config.model.positional_embedding.base,
                scale=config.model.positional_embedding.scaling_factor,
                offset=config.model.positional_embedding.offset,
            )

        model_provider_func = get_model_provider_func(config)
        self.model = model_provider_func(config)
        self.output_layernorm = get_norm(config)

        if config.model.untie_embed:
            self.output_layer = ColumnParallelLinear(
                config, config.model.d_model, tokenizer.padded_vocab_size, bias=False
            )

        self.loss_fn = loss_fn
        self.padding_start_idx = tokenizer.vocab_size

        # Initialize KV cache manager for inference
        self.kv_cache_manager = None
        if config.model.kv_cache.enabled:
            from ironcore.layers.kv_cache import KVCacheManager

            self.kv_cache_manager = KVCacheManager(config)

        self.init_weights()

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
    ):
        """
        Forward pass through language model.
        """
        input_ids = input_ids.to(self.device, non_blocking=True)
        if labels is not None:
            labels = labels.to(self.device, non_blocking=True)

        # Determine cache position
        if cache_position is None:
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

        attention_mask, computed_position_ids, loss_mask = self.get_masks_and_position_ids(
            input_ids, labels, cache_position=cache_position
        )
        if position_ids is None:
            position_ids = computed_position_ids
        else:
            position_ids = position_ids.to(self.device, non_blocking=True)

        x = self.embedding(input_ids, position_ids)

        model_out = self.model(
            x,
            attention_mask,
            self.rotary_pos_emb,
            position_ids=position_ids,
            use_cache=use_cache,
            past_key_values=past_key_values,
            kv_cache_manager=self.kv_cache_manager if not self.training else None,
            cache_position=cache_position if not self.training else None,
        )

        if use_cache or (self.kv_cache_manager is not None and not self.training):
            lm_output, new_key_values = model_out
        else:
            lm_output = model_out
            new_key_values = None

        lm_output = self.output_layernorm(lm_output)

        if self.config.model.untie_embed:
            logits_parallel = self.output_layer(lm_output)
        else:
            from ironcore.parallel.tensor_parallel import comm

            input_parallel = comm.copy_inputs_to_model_parallel_workers(lm_output)
            logits_parallel = F.linear(input_parallel, self.embedding.word_embeddings.weight)

        if labels is None:
            logits = _gather_tensor_along_last_dim(logits_parallel)
            if use_cache or (self.kv_cache_manager is not None and not self.training):
                return logits, new_key_values
            return logits

        losses = self.compute_loss_from_logits(
            logits_parallel, labels, loss_mask, self.fp16_lm_cross_entropy, self.padding_start_idx
        )
        return losses

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 128,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = 0,
        do_sample: bool = False,
        eos_token_id: int | None = None,
    ) -> torch.Tensor:
        """
        Autoregressive generation with KV cache.
        """
        batch_size = input_ids.size(0)
        generated = input_ids.clone()
        past_key_values = None
        done = torch.zeros(batch_size, dtype=torch.bool, device=input_ids.device)
        next_token = input_ids

        # Use stateful cache if enabled
        use_stateful = self.kv_cache_manager is not None
        if use_stateful:
            self.initialize_cache(batch_size, input_ids.device)

        for step in range(max_new_tokens):
            cur_input = input_ids if step == 0 else next_token
            cur_cache_pos = self.kv_cache_manager.get_cache_position() if use_stateful else None

            out = self.forward(
                cur_input,
                labels=None,
                use_cache=not use_stateful,
                past_key_values=past_key_values,
                cache_position=cur_cache_pos,
            )

            if use_stateful:
                logits, _ = out
            else:
                logits, past_key_values = out

            next_logits = logits[:, -1, :]
            if parallel_states.get_tensor_model_parallel_world_size() > 1:
                next_logits = _gather_tensor_along_last_dim(next_logits)

            next_token = self._sample(next_logits, temperature, top_p, top_k, do_sample)

            if do_sample and parallel_states.get_tensor_model_parallel_world_size() > 1:
                dist.broadcast(
                    next_token,
                    src=0,
                    group=parallel_states.get_tensor_model_parallel_group(),
                )

            if eos_token_id is not None:
                new_done = (next_token.squeeze(1) == eos_token_id) | done
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

    def get_masks_and_position_ids(self, input_ids, labels=None, cache_position=0):
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

        # Correct masking for stateful cache
        if seq_len == 1 and not isinstance(cache_position, torch.Tensor) and cache_position > 0:
            attention_mask = torch.ones(
                (att_mask_batch, 1, 1, total_len), dtype=torch.bool, device=input_ids.device
            )
        else:
            full_causal_mask = torch.tril(
                torch.ones((total_len, total_len), device=input_ids.device, dtype=torch.bool)
            )
            if not isinstance(cache_position, torch.Tensor):
                if cache_position == 0:
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

        loss_mask = torch.ones(input_ids.size(), dtype=torch.float, device=input_ids.device)
        return attention_mask, position_ids, loss_mask

    def initialize_cache(
        self, batch_size: int, device: torch.device, dtype: torch.dtype | None = None
    ):
        if self.kv_cache_manager is not None:
            self.kv_cache_manager.initialize(batch_size, len(self.model.layers), device, dtype)

    def reset_cache(self, batch_indices: list[int] | None = None):
        if self.kv_cache_manager is not None:
            self.kv_cache_manager.reset(batch_indices)

    def get_cache_statistics(self) -> dict:
        if self.kv_cache_manager is not None:
            return self.kv_cache_manager.get_statistics()
        return {"initialized": False}
