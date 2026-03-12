# Copyright (c) 2025-2026 Jaegeun Han
#
# SPDX-License-Identifier: Apache-2.0

"""Rollout utilities for GRPO with prefix KV-cache sharing.

This module provides efficient batched generation for GRPO rollouts,
expanding the KV-cache from [B] prompts to [B×G] completions after
prefill for parallel generation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.distributed as dist
import torch.nn.functional as F

from .buffer import RolloutBuffer

if TYPE_CHECKING:
    from collections.abc import Sequence


def _expand_kv_cache(
    past_key_values: Sequence[tuple[torch.Tensor, torch.Tensor]],
    group_size: int,
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Expand KV-cache from [B, ...] to [B×G, ...].

    Each prompt's KV cache is replicated G times to generate G completions.

    Args:
        past_key_values: List of (key, value) tuples per layer
            key: [B, num_heads, prompt_len, head_dim]
        group_size: Number of completions per prompt (G)

    Returns:
        Expanded KV-cache list with [B×G, num_heads, prompt_len, head_dim]
    """
    expanded_kv = []

    for layer_kv in past_key_values:
        key, value = layer_kv
        # key: [B, num_heads, prompt_len, head_dim]

        # Expand: repeat each sample G times
        # [B, num_heads, prompt_len, head_dim] -> [B*G, num_heads, prompt_len, head_dim]
        # repeat_interleave ensures contiguous layout which is safer for some backends
        expanded_key = key.repeat_interleave(group_size, dim=0)
        expanded_value = value.repeat_interleave(group_size, dim=0)

        expanded_kv.append((expanded_key, expanded_value))

    return expanded_kv


def _sample_tokens_batched(
    logits: torch.Tensor,
    temperature: float,
    top_p: float,
    top_k: int,
    do_sample: bool,
    tp_group=None,
) -> torch.Tensor:
    """Sample tokens for entire batch at once.

    Args:
        logits: [batch, vocab] unnormalized logits
        temperature: Sampling temperature
        top_p: Nucleus sampling threshold
        top_k: Top-k cutoff (0 = disabled)
        do_sample: If False, use greedy decoding
        tp_group: Tensor parallel group for broadcast (if sampling)

    Returns:
        [batch, 1] sampled token IDs
    """
    if not do_sample:
        return logits.argmax(dim=-1, keepdim=True)

    if temperature != 1.0:
        logits = logits / temperature

    # Top-k filtering
    if top_k > 0:
        top_k = min(top_k, logits.size(-1))
        # Get k-th largest value for each batch element
        kth_vals = logits.topk(top_k, dim=-1).values[:, -1].unsqueeze(-1)
        logits = logits.masked_fill(logits < kth_vals, float("-inf"))

    # Top-p (nucleus) filtering
    if top_p < 1.0:
        sorted_logits, sorted_idx = logits.sort(dim=-1, descending=True)
        sorted_probs = sorted_logits.softmax(dim=-1)
        cumprobs = sorted_probs.cumsum(dim=-1)

        # Remove tokens whose cumulative probability exceeds top_p
        # Keep the token that first pushes over the threshold
        sorted_probs_for_mask = sorted_logits.softmax(dim=-1)
        remove = cumprobs - sorted_probs_for_mask > top_p
        sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))

        # Scatter back to original order
        logits = logits.scatter(-1, sorted_idx, sorted_logits)

    # Sample from probability distribution
    probs = logits.softmax(dim=-1)
    sampled = torch.multinomial(probs, num_samples=1)

    # Under TP with stochastic sampling, broadcast from rank 0
    # (greedy is deterministic so all ranks agree without communication)
    if tp_group is not None:
        dist.broadcast(sampled, src=0, group=tp_group)

    return sampled


def _compute_token_log_probs_batched(
    logits: torch.Tensor,
    token_ids: torch.Tensor,
) -> torch.Tensor:
    """Compute log probability of token_ids from logits.

    Args:
        logits: [batch, vocab] logits
        token_ids: [batch] token IDs

    Returns:
        [batch] log probabilities
    """
    log_probs = F.log_softmax(logits.float(), dim=-1)
    return log_probs.gather(dim=-1, index=token_ids.unsqueeze(-1)).squeeze(-1)


@torch.no_grad()
def generate_rollouts_batched(
    model: torch.nn.Module,
    prompt_ids: torch.Tensor,
    group_size: int,
    metadata: list[dict],
    max_new_tokens: int = 512,
    temperature: float = 1.0,
    top_p: float = 0.9,
    top_k: int = 0,
    do_sample: bool = True,
    eos_token_id: int | None = None,
) -> RolloutBuffer:
    """Generate G completions per prompt with BATCHED prefix KV-cache.

    Algorithm:
    1. Prefill prompts [B, prompt_len] -> prefix_kv, prefill_logits
    2. Expand prefix_kv: [B, ...] -> [B×G, ...] by repeating each G times
    3. Sample first tokens from prefill_logits (batched)
    4. Autoregressively generate all B×G sequences in parallel

    Args:
        model: Language model with forward(use_cache=True) support
        prompt_ids: [B, prompt_len] prompt token IDs
        group_size: Number of completions per prompt (G)
        metadata: List of metadata dicts for each prompt [B]
        max_new_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        top_p: Nucleus sampling threshold
        top_k: Top-k cutoff
        do_sample: If False, use greedy decoding
        eos_token_id: EOS token ID for early stopping

    Returns:
        RolloutBuffer with completions, log probs, and metadata
    """
    B, prompt_len = prompt_ids.shape
    G = group_size
    device = prompt_ids.device
    total_samples = B * G

    # Initialize KV cache if model supports it
    # Handle wrapped models (DDP, FSDP)
    # Note: Initialize with B (prompt batch size), not total_samples
    # The expanded KV for group generation is handled via past_key_values
    unwrapped_model = model
    if hasattr(model, "module"):
        unwrapped_model = model.module
    if hasattr(unwrapped_model, "initialize_cache"):
        unwrapped_model.initialize_cache(B, device)

    # Get TP group if applicable
    tp_group = None
    if dist.is_initialized():
        try:
            from ironcore.parallel.parallel_states import (
                get_tensor_model_parallel_group,
                get_tensor_model_parallel_world_size,
            )

            if get_tensor_model_parallel_world_size() > 1:
                tp_group = get_tensor_model_parallel_group()
        except (AssertionError, ImportError):
            pass

    # === Step 1: Prefill all prompts ===
    prefill_logits, prefix_kv = model.forward(
        prompt_ids, labels=None, use_cache=True, past_key_values=None
    )
    # prefill_logits: [B, prompt_len, vocab]
    # prefix_kv: List of (key, value) per layer
    #   key: [B, num_heads, prompt_len, head_dim]

    # === Step 2: Expand KV-cache to [B×G, ...] ===
    expanded_kv = _expand_kv_cache(prefix_kv, G)

    # === Step 3: Sample first tokens ===
    # Get logits at last position
    last_logits = prefill_logits[:, -1, :]  # [B, vocab]

    # Expand logits for G samples per prompt: [B, vocab] -> [B*G, vocab]
    expanded_logits = last_logits.unsqueeze(1).expand(B, G, -1).reshape(total_samples, -1)

    # Sample first tokens for all B×G
    first_tokens = _sample_tokens_batched(
        expanded_logits, temperature, top_p, top_k, do_sample, tp_group
    )
    # first_tokens: [B×G, 1]

    # Compute log probs for first tokens
    first_log_probs = _compute_token_log_probs_batched(expanded_logits, first_tokens.squeeze(-1))

    # === Step 4: Autoregressive generation (batched) ===
    # Pre-allocate tensors for efficiency
    generated = torch.zeros((total_samples, max_new_tokens), dtype=torch.long, device=device)
    generated[:, 0] = first_tokens.squeeze(-1)

    log_probs_list = [first_log_probs]
    done_mask = torch.zeros(total_samples, dtype=torch.bool, device=device)
    past_kv = expanded_kv

    for t in range(1, max_new_tokens):
        if done_mask.all():
            break

        # Forward with cached KV
        logits, past_kv = model.forward(
            generated[:, t-1 : t],  # Only last token
            labels=None,
            use_cache=True,
            past_key_values=past_kv,
        )
        # logits: [B×G, 1, vocab]

        next_logits = logits[:, 0, :]  # [B×G, vocab]

        next_tokens = _sample_tokens_batched(
            next_logits, temperature, top_p, top_k, do_sample, tp_group
        )
        # next_tokens: [B×G, 1]

        # Compute log probs
        next_log_probs = _compute_token_log_probs_batched(next_logits, next_tokens.squeeze(-1))

        # Mask log probs for sequences that are already done
        if eos_token_id is not None:
            log_probs_list.append(next_log_probs * (~done_mask).float())
            done_mask = done_mask | (next_tokens.squeeze(-1) == eos_token_id)
        else:
            log_probs_list.append(next_log_probs)

        generated[:, t] = next_tokens.squeeze(-1)

    # Trim to actual length if all sequences reached EOS early
    actual_len = len(log_probs_list)
    generated = generated[:, :actual_len]

    # === Step 5: Build output ===
    # Expand prompt_ids to [B×G, prompt_len]
    expanded_prompts = prompt_ids.unsqueeze(1).expand(B, G, -1).reshape(total_samples, prompt_len)

    # Concatenate prompts + completions
    completion_ids = torch.cat([expanded_prompts, generated], dim=1)  # [B×G, total_len]

    # Compute total log probs per sequence
    log_probs_stacked = torch.stack(log_probs_list, dim=1)  # [B×G, gen_len]
    old_log_probs = log_probs_stacked.sum(dim=1)  # [B×G]

    # Group IDs: [0,0,0,0, 1,1,1,1, ...]
    group_ids = torch.arange(B, device=device).unsqueeze(1).expand(B, G).reshape(-1)

    # Expand metadata
    expanded_metadata = []
    for meta in metadata:
        expanded_metadata.extend([meta.copy() for _ in range(G)])

    return RolloutBuffer(
        prompt_ids=prompt_ids,
        prompt_attention_mask=torch.ones_like(prompt_ids),
        completion_ids=completion_ids,
        response_ids=generated,
        old_log_probs=old_log_probs,
        rewards=torch.zeros(total_samples, device=device),
        advantages=torch.zeros(total_samples, device=device),
        group_ids=group_ids,
        metadata=expanded_metadata,
    )


# Alias for backward compatibility
generate_rollouts_with_prefix_cache = generate_rollouts_batched
