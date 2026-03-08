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
    vocab_parallel_cross_entropy,
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
        self, input_ids, labels=None, position_ids=None, use_cache=False, past_key_values=None
    ):
        """
        Forward pass through language model.

        Args:
            input_ids: [b, s] Input token IDs
            labels: [b, s] Target token IDs (for training)
            position_ids: [b, s] Optional position IDs (for bin-packed sequences)
            use_cache: Whether to use KV cache
            past_key_values: List of past (key, value) tuples for each layer

        Returns:
            If use_cache and labels is None: (logits, new_key_values)
            Otherwise: outputs (logits or loss)
        """
        input_ids = input_ids.to(self.device, non_blocking=True)
        if labels is not None:
            labels = labels.to(self.device, non_blocking=True)

        # Determine cache position
        cache_position = 0
        if use_cache and past_key_values is not None and len(past_key_values) > 0:
            # Get cache length from first layer's past key
            # Validate past_key_values structure before accessing
            first_layer_kv = past_key_values[0]
            if (
                isinstance(first_layer_kv, tuple | list)
                and len(first_layer_kv) >= 2
                and first_layer_kv[0] is not None
            ):
                past_key = first_layer_kv[0]
                if past_key.dim() >= 2:
                    cache_position = past_key.size(1)
                else:
                    raise ValueError(
                        f"Invalid past_key_values shape: expected at least 2D, got {past_key.dim()}D"
                    )
            else:
                raise ValueError(
                    "Invalid past_key_values structure: expected tuple/list of (key, value) tensors"
                )

        attention_mask, computed_position_ids, loss_mask = self.get_masks_and_position_ids(
            input_ids, labels, cache_position=cache_position
        )
        # Use provided position_ids if available (for bin-packed sequences), otherwise use computed ones
        if position_ids is None:
            position_ids = computed_position_ids
        else:
            position_ids = position_ids.to(self.device, non_blocking=True)

        # input_ids: [b s]
        # attention_mask: [b, 1, s, s]
        # position_ids: [b, s]
        # loss_mask: [b s]

        # pre process
        # x: [b, s, h]
        x = self.embedding(input_ids, position_ids)

        model_out = self.model(
            x,
            attention_mask,
            self.rotary_pos_emb,
            position_ids=position_ids,
            use_cache=use_cache,
            past_key_values=past_key_values,
        )

        # Handle cache output
        if use_cache:
            lm_output, new_key_values = model_out
        else:
            lm_output = model_out

        # layer norm
        lm_output = self.output_layernorm(lm_output)

        # post process
        # lm_output: [b, s, h]
        outputs = self.post_lm_processing(
            lm_output,
            labels,
            loss_mask,
            self.fp16_lm_cross_entropy,
            padding_start_idx=self.padding_start_idx,
        )

        # outputs: logits[b, s, v] or loss[b, s]
        if use_cache and labels is None:
            return outputs, new_key_values
        return outputs

    def _forward_inference_with_cache(self, input_ids, cache_position=0):
        """Inference forward pass using stateful KVCacheManager.

        Args:
            input_ids: [b, s] Input token IDs
            cache_position: Starting position in cache (int or [b] tensor)

        Returns:
            logits [b, s, vocab]
        """
        input_ids = input_ids.to(self.device, non_blocking=True)

        attention_mask, position_ids, _ = self.get_masks_and_position_ids(
            input_ids, None, cache_position=cache_position
        )

        # Embedding
        x = self.embedding(input_ids, position_ids)

        # Use stateful cache path through model
        lm_output = self.model._forward_with_cache_manager(
            x,
            attention_mask,
            self.rotary_pos_emb,
            self.kv_cache_manager,
            cache_position,
            position_ids=position_ids,
        )

        # Layer norm
        lm_output = self.output_layernorm(lm_output)

        # Post process (no labels for inference)
        outputs = self.post_lm_processing(
            lm_output,
            None,
            None,
            self.fp16_lm_cross_entropy,
            padding_start_idx=self.padding_start_idx,
        )

        return outputs

    def get_masks_and_position_ids(self, input_ids, labels=None, cache_position=0):
        """
        Get attention masks and position IDs.

        Args:
            input_ids: [b, s] Input token IDs
            labels: Optional labels for loss masking
            cache_position: Starting position for position IDs (int or [b] tensor)

        Returns:
            Tuple of (attention_mask, position_ids, loss_mask)
        """
        # attention mask (lower triangular)
        if input_ids.dim() == 2:
            att_mask_batch = input_ids.size(0)  # micro_batch_size
        else:
            att_mask_batch = 1

        seq_len = input_ids.size(1)

        if isinstance(cache_position, torch.Tensor):
            # Per-sequence positions (continuous batching)
            max_cache_pos = cache_position.max().item()
            total_len = int(max_cache_pos + seq_len)

            # Position IDs
            # position_ids: [b, s]
            position_ids = cache_position.unsqueeze(1) + torch.arange(
                seq_len, device=input_ids.device
            )
        else:
            # Uniform cache position
            total_len = int(cache_position + seq_len)
            position_ids = (
                torch.arange(
                    cache_position,
                    cache_position + seq_len,
                    dtype=torch.long,
                    device=input_ids.device,
                )
                .unsqueeze(0)
                .expand(att_mask_batch, seq_len)
            )

        # Create causal mask for the context
        # Simplified: for inference with cache_position > 0, we typically only have 1 token
        # attending to all past tokens.
        if seq_len == 1 and not isinstance(cache_position, torch.Tensor):
            # Optimized path for single token generation (uniform positions)
            attention_mask = torch.ones(
                (att_mask_batch, 1, 1, total_len),
                dtype=torch.bool,
                device=input_ids.device,
            )
        else:
            # Full causal mask for [new_tokens] attending to [cached + new]
            # [total_len, total_len]
            full_causal_mask = torch.tril(
                torch.ones(
                    (total_len, total_len),
                    device=input_ids.device,
                    dtype=torch.bool,
                )
            )

            if isinstance(cache_position, torch.Tensor):
                # Complex case: per-sequence lengths (continuous batching)
                # Each sequence b attends to its own position_ids[b, :] + past
                # position_ids shape: [batch, seq_len]
                # We want mask[b, 0, i, j] = (position_ids[b, i] >= j)

                q_pos = position_ids.unsqueeze(-1)  # [batch, seq_len, 1]
                kv_pos = torch.arange(total_len, device=input_ids.device).view(
                    1, 1, -1
                )  # [1, 1, total_len]

                # [batch, seq_len, total_len]
                attention_mask = (q_pos >= kv_pos).unsqueeze(1)  # [batch, 1, seq_len, total_len]
            else:
                # Uniform positions
                if cache_position == 0:
                    # Training case: create full batched causal mask
                    attention_mask = torch.tril(
                        torch.ones(
                            (att_mask_batch, seq_len, seq_len),
                            device=input_ids.device,
                            dtype=torch.bool,
                        )
                    ).view(att_mask_batch, 1, seq_len, seq_len)
                else:
                    # Inference with cache: slice from full causal mask
                    attention_mask = full_causal_mask[cache_position:total_len, :total_len]
                    attention_mask = attention_mask.view(att_mask_batch, 1, seq_len, total_len)

        # loss mask
        loss_mask = torch.ones(input_ids.size(), dtype=torch.float, device=input_ids.device)
        if self.eod_mask_loss and labels is not None:
            loss_mask[labels == get_tokenizer().eos_token_id] = 0
            loss_mask[labels == get_tokenizer().pad_token_id] = 0

        # Resetting logic for packed sequences (training with multiple documents)
        # Only apply when cache_position is 0 (no KV cache, i.e., training or prefill)
        if (self.reset_position_ids or self.reset_attention_mask) and cache_position == 0:
            # loop through the batches
            for b in range(input_ids.size(0)):
                # find indices of EOD tokens in input_ids
                eod_indices = (input_ids[b] == get_tokenizer().eod_token_id).nonzero(as_tuple=True)[0]
                # detach indices from position if going to modify them
                if self.reset_position_ids:
                    eod_indices = eod_indices.clone()

                # reset position ids along with EOD indices
                prev_index = 0
                for j in range(eod_indices.size(0)):
                    i = eod_indices[j].item()
                    # reset attention mask: tokens after EOD can't attend to tokens before EOD
                    if self.reset_attention_mask:
                        attention_mask[b, 0, (i + 1) :, : (i + 1)] = False
                    # reset position: restart counting after EOD
                    if self.reset_position_ids:
                        position_ids[b, (i + 1) :] -= i + 1 - prev_index
                        prev_index = i + 1

        return attention_mask, position_ids, loss_mask

    def compute_loss_from_logits(
        self, logits, labels, loss_mask, fp16_lm_cross_entropy=False, padding_start_idx: int = None
    ):
        """Compute loss from logits using vocab_parallel_cross_entropy.

        This method is reusable for both training and evaluation.
        It handles both TP=1 and TP>1 cases correctly.

        Args:
            logits: [batch, seq_len, vocab_size] or [batch, seq_len, vocab_size/tp]
            labels: [batch, seq_len] ground truth token IDs
            loss_mask: [batch, seq_len] valid token mask
            fp16_lm_cross_entropy: Whether to use fp16 for cross entropy
            padding_start_idx: Index where padding tokens start in vocab

        Returns:
            Scalar loss value
        """
        # Ensure labels are contiguous
        labels = labels.contiguous()

        # Convert to appropriate dtype
        if fp16_lm_cross_entropy:
            logits = logits.to(dtype=torch.half)
        else:
            logits = logits.float()

        # Compute per-token losses using vocab_parallel_cross_entropy
        # This handles both TP=1 and TP>1 cases correctly
        per_token_losses = vocab_parallel_cross_entropy(
            vocab_parallel_logits=logits,
            labels=labels,
            padding_start_idx=padding_start_idx,
        ).contiguous()

        # Apply loss function (e.g., loss_func_sft for per-sample averaging)
        loss = self.loss_fn(per_token_losses, loss_mask)

        return loss

    def post_lm_processing(
        self,
        lm_output,
        labels,
        loss_mask,
        fp16_lm_cross_entropy=False,
        padding_start_idx: int = None,
    ):
        # b: batch size
        # s: sequence length
        # h: hidden_size
        # v: vocab_size
        # mp: tensor model parallel size

        # Compute logits from model output
        # If weights are tied, embedding is VocabParallel (RowParallel style: shard rows/vocab)
        # If weights are untied, output_layer is ColumnParallel (shard columns/vocab)
        # In both cases, matmul with lm_output (unsharded) results in vocab-parallel logits.

        if self.config.model.untie_embed:
            logits_parallel = torch.matmul(lm_output, self.output_layer.weight)
        else:
            logits_parallel = torch.matmul(
                lm_output, self.embedding.word_embeddings.weight.transpose(0, 1)
            )

        if labels is None:
            # Return full logits for inference/evaluation
            # Gather from all TP ranks to get full vocab
            from ironcore.parallel.tensor_parallel import comm

            logits = comm.gather_from_model_parallel_workers(
                logits_parallel, {"column_parallel": True, "concatenated_weights": 1}
            )
            return logits

        # Compute loss from logits using shared method
        # This function expects parallelized logits
        losses = self.compute_loss_from_logits(
            logits_parallel, labels, loss_mask, fp16_lm_cross_entropy, padding_start_idx
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

        Prefills the prompt in a single forward pass, then decodes one token
        at a time using cached K/V. Works under tensor parallelism.

        Args:
            input_ids: [batch, prompt_len] prompt token IDs
            max_new_tokens: Maximum number of tokens to generate
            temperature: Softmax temperature (<1.0 = sharper, >1.0 = flatter)
            top_p: Nucleus sampling threshold (1.0 = disabled)
            top_k: Top-k cutoff (0 = disabled)
            do_sample: If False, use greedy decoding
            eos_token_id: Stop when all sequences produce this token

        Returns:
            [batch, prompt_len + generated_len] full token IDs including prompt
        """
        batch_size = input_ids.size(0)
        generated = input_ids.clone()
        past_key_values = None
        done = torch.zeros(batch_size, dtype=torch.bool, device=input_ids.device)
        next_token = input_ids  # placeholder; overwritten before decode step uses it

        for step in range(max_new_tokens):
            # Prefill on step 0, decode one token at a time afterwards
            cur_input = input_ids if step == 0 else next_token

            logits, past_key_values = self.forward(
                cur_input,
                labels=None,
                use_cache=True,
                past_key_values=past_key_values,
            )

            # Extract logits at the last position: [batch, vocab(/tp)]
            next_logits = logits[:, -1, :]

            # Under TP, vocab dim is sharded — gather to full vocab on all ranks
            if parallel_states.get_tensor_model_parallel_world_size() > 1:
                next_logits = _gather_tensor_along_last_dim(next_logits)

            next_token = self._sample(next_logits, temperature, top_p, top_k, do_sample)

            # Under TP with stochastic sampling, synchronize token from rank 0
            # (greedy is deterministic so all ranks agree without communication)
            if do_sample and parallel_states.get_tensor_model_parallel_world_size() > 1:
                dist.broadcast(
                    next_token,
                    src=0,
                    group=parallel_states.get_tensor_model_parallel_group(),
                )

            generated = torch.cat([generated, next_token], dim=1)

            if eos_token_id is not None:
                done = done | (next_token.squeeze(-1) == eos_token_id)
                if done.all():
                    break

        return generated

    def _sample(
        self,
        logits: torch.Tensor,
        temperature: float,
        top_p: float,
        top_k: int,
        do_sample: bool,
    ) -> torch.Tensor:
        """
        Sample next token from logits.

        Args:
            logits: [batch, vocab] unnormalized logits
            temperature: Sampling temperature
            top_p: Nucleus sampling threshold
            top_k: Top-k cutoff
            do_sample: If False, return argmax

        Returns:
            [batch, 1] next token IDs
        """
        if not do_sample:
            return logits.argmax(dim=-1, keepdim=True)

        if temperature != 1.0:
            logits = logits / temperature

        # Top-k: zero out all but the top-k logits
        if top_k > 0:
            top_k = min(top_k, logits.size(-1))
            kth_vals = logits.topk(top_k, dim=-1).values[:, -1, None]
            logits = logits.masked_fill(logits < kth_vals, float("-inf"))

        # Top-p (nucleus): zero out tokens outside the nucleus
        if top_p < 1.0:
            sorted_logits, sorted_idx = logits.sort(dim=-1, descending=True)
            cumprobs = sorted_logits.softmax(dim=-1).cumsum(dim=-1)
            # Remove tokens whose cumulative probability exceeds top_p,
            # but keep the token that first pushes over the threshold
            remove = (cumprobs - sorted_logits.softmax(dim=-1)) > top_p
            sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))
            logits = torch.full_like(logits, float("-inf")).scatter_(1, sorted_idx, sorted_logits)

        probs = logits.softmax(dim=-1)
        return torch.multinomial(probs, num_samples=1)

    def _should_use_stateful_cache(self) -> bool:
        """Check if stateful cache should be used.
        Returns:
            True if not in training mode and kv_cache_manager exists and and is initialized
        """
        return (
            not self.training
            and self.kv_cache_manager is not None
            and self.kv_cache_manager.is_initialized
        )

    def initialize_cache(
        self, batch_size: int, device: torch.device, dtype: torch.dtype | None = None
    ):
        """Initialize KV cache for inference.
        Args:
            batch_size: Number of sequences in batch
            device: Device to allocate cache on
            dtype: Data type for cache (defaults to model dtype)
        """
        if self.kv_cache_manager is None:
            return
        self.kv_cache_manager.initialize(
            batch_size=batch_size,
            num_layers=self.config.model.num_layers,
            device=device,
            dtype=dtype,
        )

    def reset_cache(self, batch_indices: list[int] | None = None):
        """Reset KV cache for specified sequences.
        Args:
            batch_indices: Indices of sequences to reset. If None, reset all.
        """
        if self.kv_cache_manager is None:
            return
        self.kv_cache_manager.reset(batch_indices)

    def get_cache_statistics(self) -> dict:
        """Get cache statistics for monitoring.
        Returns:
            Dictionary with cache statistics
        """
        if self.kv_cache_manager is None:
            return {"initialized": False}
        return self.kv_cache_manager.get_statistics()
