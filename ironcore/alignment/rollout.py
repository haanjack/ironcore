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

from ironcore.parallel import parallel_states
from ironcore.utils import profile_context

from .buffer import RolloutBuffer

if TYPE_CHECKING:
    from collections.abc import Sequence


def _build_rollout_output(
    prompt_ids: torch.Tensor,
    generated: torch.Tensor,
    log_probs_list: list[torch.Tensor],
    response_lengths: torch.Tensor,
    group_size: int,
    metadata: list[dict],
) -> RolloutBuffer:
    """Build RolloutBuffer from generation outputs (shared by both rollout paths)."""
    B, prompt_len = prompt_ids.shape
    G = group_size
    total_samples = B * G
    device = prompt_ids.device

    actual_len = len(log_probs_list)
    generated = generated[:, :actual_len]
    response_lengths = response_lengths.clamp(max=actual_len)

    expanded_prompts = prompt_ids.unsqueeze(1).expand(B, G, -1).reshape(total_samples, prompt_len)
    completion_ids = torch.cat([expanded_prompts, generated], dim=1)

    log_probs_stacked = torch.stack(log_probs_list, dim=1)
    old_log_probs = log_probs_stacked.sum(dim=1)

    # Offset by dp_rank * B so group_ids are globally unique across the data-parallel
    # group. compute_advantages() all-gathers rewards/group_ids across ranks and
    # normalizes per unique group_id; without this offset, group 0 on rank 0 and
    # group 0 on rank 1 (different prompts) would be pooled together.
    dp_rank = parallel_states.get_data_parallel_group_rank()
    group_ids = torch.arange(B, device=device).unsqueeze(1).expand(B, G).reshape(-1) + dp_rank * B

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
        response_lengths=response_lengths,
    )


def _fsdp_done_check(done_mask: torch.Tensor, device: torch.device) -> bool | None:
    """Coordinate decode loop termination across FSDP data-parallel ranks.

    FSDP requires all ranks to call model.forward() the same number of times.
    When one rank's sequences finish before another's, the finished rank must
    keep issuing dummy forwards until all ranks are done.

    Must be called every decode step by ALL ranks simultaneously so the
    underlying dist.all_reduce collective is always balanced.

    Returns:
        True  — all FSDP ranks are done; break the loop.
        False — this rank is done but at least one other rank isn't; do a dummy forward.
        None  — this rank still has active sequences; proceed normally.
    """
    locally_active = not done_mask.all()

    if not dist.is_initialized():
        return None if locally_active else True

    # All ranks participate every step: 1 = still active, 0 = done.
    active_signal = torch.tensor(1 if locally_active else 0, dtype=torch.int8, device=device)
    dist.all_reduce(active_signal, op=dist.ReduceOp.MAX)

    any_rank_active = active_signal.item() > 0
    if not any_rank_active:
        return True  # Every rank finished — safe to break.
    if not locally_active:
        return False  # This rank is done but others aren't — dummy forward.
    return None  # This rank still has active sequences.


def _expand_kv_cache(
    past_key_values: Sequence[tuple[torch.Tensor, torch.Tensor]],
    group_size: int,
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Expand KV-cache from [B, ...] to [B×G, ...].

    Each prompt's KV cache is replicated G times to generate G completions.

    Args:
        past_key_values: List of (key, value) tuples per layer
            key: [B, prompt_len, num_heads, head_dim]
        group_size: Number of completions per prompt (G)

    Returns:
        Expanded KV-cache list with [B×G, prompt_len, num_heads, head_dim]
    """
    expanded_kv = []

    for layer_kv in past_key_values:
        key, value = layer_kv
        # key: [B, prompt_len, num_heads, head_dim]

        # Optimization: repeat each sample G times.
        # repeat_interleave ensures contiguous layout which is safer for optimized attention kernels.
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
        remove = cumprobs - sorted_probs > top_p
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


def _filter_logits(
    logits: torch.Tensor,
    temperature: float,
    top_p: float,
    top_k: int,
) -> torch.Tensor:
    """Apply the same temperature / top-k / top-p filtering used at sample time.

    ``_compute_token_log_probs_batched`` MUST be called on these filtered
    logits so the recorded ``old_log_probs`` correspond to the behaviour
    policy that actually generated the tokens (temperature-scaled, top-p
    filtered). Using raw logits makes the IS-ratio denominator describe a
    different distribution. (Fable issue #70.)
    """
    if temperature != 1.0:
        logits = logits / temperature
    if top_k > 0:
        top_k = min(top_k, logits.size(-1))
        kth_vals = logits.topk(top_k, dim=-1).values[:, -1].unsqueeze(-1)
        logits = logits.masked_fill(logits < kth_vals, float("-inf"))
    if top_p < 1.0:
        sorted_logits, sorted_idx = logits.sort(dim=-1, descending=True)
        sorted_probs = sorted_logits.softmax(dim=-1)
        cumprobs = sorted_probs.cumsum(dim=-1)
        remove = cumprobs - sorted_probs > top_p
        sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))
        logits = logits.scatter(-1, sorted_idx, sorted_logits)
    return logits


def _compute_token_log_probs_batched(
    logits: torch.Tensor,
    token_ids: torch.Tensor,
) -> torch.Tensor:
    """Compute log probability of token_ids from logits.

    Args:
        logits: [batch, vocab] logits (already temperature/top-p/top-k filtered
            by the sampling path — see _filter_logits)
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
    if (
        dist.is_initialized()
        and parallel_states.is_model_parallel_initialized()
        and parallel_states.get_tensor_model_parallel_world_size() > 1
    ):
        tp_group = parallel_states.get_tensor_model_parallel_group()

    # === Step 1: Prefill all prompts ===
    with profile_context("grpo_prefill"):
        prefill_logits, prefix_kv = model.forward(
            prompt_ids, labels=None, use_cache=True, past_key_values=None
        )
    # prefill_logits: [B, prompt_len, vocab]
    # prefix_kv: List of (key, value) per layer
    #   key: [B, prompt_len, num_heads, head_dim]

    # === Step 2: Expand KV-cache to [B×G, ...] ===
    with profile_context("grpo_kv_expand"):
        expanded_kv = _expand_kv_cache(prefix_kv, G)

    # === Step 3: Sample first tokens ===
    with profile_context("grpo_sampling"):
        # Get logits at last position
        last_logits = prefill_logits[:, -1, :]  # [B, vocab]

        # Expand logits for G samples per prompt: [B, vocab] -> [B*G, vocab]
        expanded_logits = last_logits.unsqueeze(1).expand(B, G, -1).reshape(total_samples, -1)

        # Sample first tokens for all B×G
        first_tokens = _sample_tokens_batched(
            expanded_logits, temperature, top_p, top_k, do_sample, tp_group
        )
        # first_tokens: [B×G, 1]

        # Compute log probs for first tokens — use the SAME filtered logits
        # that sampling used, so old_log_probs describe the behaviour policy.
        # (Fable issue #70.)
        filtered_logits = _filter_logits(expanded_logits, temperature, top_p, top_k)
        first_log_probs = _compute_token_log_probs_batched(
            filtered_logits, first_tokens.squeeze(-1)
        )

    # === Step 4: Autoregressive generation (batched) ===
    # Pre-allocate tensors for efficiency
    generated = torch.zeros((total_samples, max_new_tokens), dtype=torch.long, device=device)
    generated[:, 0] = first_tokens.squeeze(-1)

    log_probs_list = [first_log_probs]
    done_mask = torch.zeros(total_samples, dtype=torch.bool, device=device)
    # response_lengths[i] = number of tokens generated for sequence i (EOS inclusive, padding exclusive)
    response_lengths = torch.full((total_samples,), max_new_tokens, dtype=torch.long, device=device)
    past_kv = expanded_kv

    # Check if any first tokens are EOS
    if eos_token_id is not None:
        newly_done = first_tokens.squeeze(-1) == eos_token_id
        response_lengths[newly_done] = 1
        done_mask = done_mask | newly_done

    with profile_context("grpo_decode_loop"):
        # When using FSDP with multiple ranks, different prompts can produce
        # different generation lengths.  Without synchronisation the rank that
        # finishes early skips FSDP all-gathers, deadlocking the other rank.
        # We communicate done status via all_reduce on the default process group
        # (separate from FSDP's PG) before each forward, so both ranks always
        # call forward the same number of times.

        for t in range(1, max_new_tokens):
            done_result = _fsdp_done_check(done_mask, device)
            if done_result is True:
                break
            if done_result is False:
                # Dummy forward to keep FSDP collective ordering in sync
                model.forward(
                    generated[:, t - 1 : t],
                    labels=None,
                    use_cache=True,
                    past_key_values=past_kv,
                )
                continue

            # Forward with cached KV
            logits, past_kv = model.forward(
                generated[:, t - 1 : t],  # Only last token
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

            # Compute log probs from filtered logits matching the sampling
            # distribution. (Fable issue #70.)
            filtered_next = _filter_logits(next_logits, temperature, top_p, top_k)
            next_log_probs = _compute_token_log_probs_batched(
                filtered_next, next_tokens.squeeze(-1)
            )

            # Mask log probs for sequences that are already done
            if eos_token_id is not None:
                newly_done = (~done_mask) & (next_tokens.squeeze(-1) == eos_token_id)
                response_lengths[newly_done] = t + 1  # +1: EOS token is included
                log_probs_list.append(next_log_probs * (~done_mask).float())
                done_mask = done_mask | newly_done
            else:
                log_probs_list.append(next_log_probs)

            generated[:, t] = next_tokens.squeeze(-1)

    return _build_rollout_output(
        prompt_ids,
        generated,
        log_probs_list,
        response_lengths,
        group_size,
        metadata,
    )


# Alias for backward compatibility
generate_rollouts_with_prefix_cache = generate_rollouts_batched


@torch.no_grad()
def generate_rollouts_paged(
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
    prompt_lengths: torch.Tensor | None = None,
) -> RolloutBuffer:
    """Generate G completions per prompt using block-based paged KV cache.

    Algorithm:
    1. Prefill each prompt into block cache (seq_id=0..B-1)
    2. Share prefix blocks: for each prompt, replicate block table to G destinations
       (O(B * num_prefix_blocks) metadata ops, no tensor copies)
    3. Autoregressively decode all B×G sequences in parallel
    4. Free all sequences

    Args:
        model: Language model with block_kv_cache_manager support
        prompt_ids: [B, prompt_len] prompt token IDs. May be padded; pass
            ``prompt_lengths`` to avoid caching padding tokens.
        group_size: Number of completions per prompt (G)
        metadata: List of metadata dicts for each prompt [B]
        max_new_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        top_p: Nucleus sampling threshold
        top_k: Top-k cutoff
        do_sample: If False, use greedy decoding
        eos_token_id: EOS token ID for early stopping
        prompt_lengths: Optional [B] int tensor of true (unpadded) prompt lengths.
            When provided, each prompt is sliced to its true length before prefill
            so padding tokens are never written into the KV cache. RoPE positions
            are therefore computed over the actual token sequence only.

    Returns:
        RolloutBuffer with completions, log probs, and metadata
    """
    B, prompt_len = prompt_ids.shape
    G = group_size
    device = prompt_ids.device
    total_samples = B * G

    unwrapped_model = model
    if hasattr(model, "module"):
        unwrapped_model = model.module

    # Switch to eval mode so block_kv_cache_manager path is active in forward()
    was_training = model.training
    if was_training:
        model.eval()

    try:
        bkv = unwrapped_model.block_kv_cache_manager
        if bkv is None or not bkv.is_initialized:
            raise RuntimeError(
                "Block KV cache not available. Set kv_cache.use_paged=true in config."
            )

        # Get TP group if applicable
        tp_group = None
        if (
            dist.is_initialized()
            and parallel_states.is_model_parallel_initialized()
            and parallel_states.get_tensor_model_parallel_world_size() > 1
        ):
            tp_group = parallel_states.get_tensor_model_parallel_group()

        block_size = bkv.block_size

        # === Step 1: Prefill each prompt into block cache ===
        # Prompts are written at their true (unpadded) length so RoPE positions
        # match training-time recomputation exactly. When prompt_lengths is given,
        # each prompt is sliced before the forward pass — padding tokens are never
        # written into the KV cache. Partial last blocks are handled correctly by
        # tokens_written tracking in gather functions.
        with profile_context("grpo_paged_prefill"):
            prefill_logits_list = []
            for i in range(B):
                true_len = (
                    int(prompt_lengths[i].item()) if prompt_lengths is not None else prompt_len
                )
                blocks_needed = (true_len + block_size - 1) // block_size
                bkv.allocate_blocks(seq_id=i, count=blocks_needed)

                single_prompt = prompt_ids[i : i + 1, :true_len]
                logits, _ = model(single_prompt, labels=None, seq_id=i)

                bkv.advance_position(seq_id=i, tokens=true_len)

                prefill_logits_list.append(logits[:, -1, :])

            prefill_logits = torch.cat(prefill_logits_list, dim=0)

        # === Step 2: Share prefix blocks ===
        # Completion seq_ids live at [B, B+total_samples) so they don't collide
        # with prompt seq_ids [0, B).
        completion_seq_ids = list(range(B, B + total_samples))
        with profile_context("grpo_paged_share"):
            for i in range(B):
                src = i
                dsts = list(range(B + i * G, B + i * G + G))
                unwrapped_model.share_prefix_cache(src, dsts)

        # === Step 3: Sample first tokens (batched) ===
        with profile_context("grpo_paged_sampling"):
            expanded_logits = (
                prefill_logits.unsqueeze(1).expand(B, G, -1).reshape(total_samples, -1)
            )
            first_tokens = _sample_tokens_batched(
                expanded_logits, temperature, top_p, top_k, do_sample, tp_group
            )
            filtered_prefill = _filter_logits(expanded_logits, temperature, top_p, top_k)
            first_log_probs = _compute_token_log_probs_batched(
                filtered_prefill, first_tokens.squeeze(-1)
            )

        # === Step 4: Autoregressive decode (all B×G in parallel) ===
        generated = torch.zeros((total_samples, max_new_tokens), dtype=torch.long, device=device)
        generated[:, 0] = first_tokens.squeeze(-1)

        log_probs_list = [first_log_probs]
        done_mask = torch.zeros(total_samples, dtype=torch.bool, device=device)
        response_lengths = torch.full(
            (total_samples,), max_new_tokens, dtype=torch.long, device=device
        )

        if eos_token_id is not None:
            newly_done = first_tokens.squeeze(-1) == eos_token_id
            response_lengths[newly_done] = 1
            done_mask = done_mask | newly_done

        with profile_context("grpo_paged_decode_loop"):
            for t in range(1, max_new_tokens):
                done_result = _fsdp_done_check(done_mask, device)
                if done_result is True:
                    break
                if done_result is False:
                    # Dummy forward: use full batch with all completion seq_ids
                    # to keep FSDP param-gather shapes consistent across ranks.
                    # Done sequences write into cache but results are ignored.
                    model.forward(
                        generated[:, t - 1 : t],
                        labels=None,
                        seq_id=completion_seq_ids,
                    )
                    continue

                cur_tokens = generated[:, t - 1 : t]

                # Forward the full batch to keep FSDP param-gather shapes
                # consistent across ranks, but only advance cache for active.
                logits, _ = model.forward(cur_tokens, labels=None, seq_id=completion_seq_ids)

                active_seq_ids = [
                    completion_seq_ids[i] for i in range(total_samples) if not done_mask[i]
                ]
                if active_seq_ids:
                    unwrapped_model.advance_cache_position(active_seq_ids, 1)

                # Mask out logits for done sequences (they attended to stale KV)
                active_mask = ~done_mask
                next_logits = logits[:, 0, :].clone()
                next_logits[~active_mask] = 0.0

                next_tokens = _sample_tokens_batched(
                    next_logits, temperature, top_p, top_k, do_sample, tp_group
                )

                filtered_next = _filter_logits(next_logits, temperature, top_p, top_k)
                next_log_probs = _compute_token_log_probs_batched(
                    filtered_next, next_tokens.squeeze(-1)
                )

                # Always mask log probs for done sequences
                log_probs_list.append(next_log_probs * active_mask.float())

                if eos_token_id is not None:
                    newly_done = active_mask & (next_tokens.squeeze(-1) == eos_token_id)
                    response_lengths[newly_done] = t + 1
                    done_mask = done_mask | newly_done

                generated[:, t] = next_tokens.squeeze(-1)

        # Free all sequences before building output
        for sid in completion_seq_ids:
            unwrapped_model.free_sequence_cache(sid)
        for sid in range(B):
            unwrapped_model.free_sequence_cache(sid)

        return _build_rollout_output(
            prompt_ids,
            generated,
            log_probs_list,
            response_lengths,
            group_size,
            metadata,
        )
    finally:
        if was_training:
            model.train()
