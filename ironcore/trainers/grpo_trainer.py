# Copyright (c) 2025-2026 Jaegeun Han
#
# SPDX-License-Identifier: Apache-2.0

"""GRPO (Group Relative Policy Optimization) Trainer.

Reference:
    DeepSeek-AI et al., "DeepSeekMath: Pushing the Limits of Mathematical
    Reasoning in Open Language Models" (2024)
    https://arxiv.org/abs/2402.03300
"""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING

import torch
from torch import distributed as dist
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

from ironcore import get_tokenizer
from ironcore.alignment.buffer import RolloutBuffer
from ironcore.alignment.dataset import get_grpo_data_iterator
from ironcore.alignment.loss.dpo import _compute_log_softmax_tp_safe, _extract_logps_from_log_probs
from ironcore.alignment.loss.grpo import compute_advantages, grpo_loss
from ironcore.alignment.loss.kl import kl_divergence_approx
from ironcore.alignment.rewards import RewardWorkerPool, get_reward_function
from ironcore.alignment.rollout import generate_rollouts_batched
from ironcore.global_vars import log_metric
from ironcore.utils import is_first_rank

from .base_trainer import BaseTrainer

if TYPE_CHECKING:
    from collections.abc import Iterator


class GRPOTrainer(BaseTrainer):
    """Trainer for Group Relative Policy Optimization (GRPO).

    GRPO improves reasoning by:
    1. Generating multiple completions per prompt (online rollout)
    2. Computing group-relative advantages
    3. Optimizing with policy gradient + KL penalty

    Key differences from DPO:
    - Online: generates completions from current policy
    - Group-based: normalizes rewards within groups
    - Reward-agnostic: works with verifiable rewards or reward models

    Attributes:
        group_size: Number of completions per prompt (G)
        beta: KL penalty coefficient
        eps: Advantage normalization epsilon
        reference_model: Frozen reference model
        reward_worker: Worker pool for reward computation
    """

    def __init__(self, config, forward_step_func, loss_fn):
        super().__init__(config, forward_step_func, loss_fn)

        # GRPO hyperparameters
        self.group_size = config.alignment.grpo_group_size
        self.beta = config.alignment.grpo_beta
        self.eps = config.alignment.grpo_eps
        self.num_epochs = config.alignment.grpo_num_epochs
        self.clip_eps = config.alignment.grpo_clip_eps
        self.rollout_chunks = config.alignment.grpo_rollout_chunks

        # Generation config
        gen_config = config.alignment.generation
        self.gen_kwargs = {
            "max_new_tokens": gen_config.max_new_tokens,
            "temperature": gen_config.temperature,
            "top_p": gen_config.top_p,
            "top_k": gen_config.top_k,
            "do_sample": gen_config.do_sample,
        }
        self.use_chat_template = gen_config.use_chat_template
        self.system_prompt = gen_config.system_prompt

        # Reward worker (will be initialized after checkpoint load)
        self.reward_worker: RewardWorkerPool | None = None
        self._reward_config = config.alignment.reward

        # Reference model (created after checkpoint load)
        self.reference_model: nn.Module | None = None

        # Tokenizer
        self._tokenizer = get_tokenizer()

        # For metrics tracking (set during train_step)
        self._current_response_lengths: torch.Tensor | None = None
        self._current_responses_text: list[str] | None = None

        self.logger.info(
            f"GRPOTrainer initialized with group_size={self.group_size}, "
            f"beta={self.beta}, gen_kwargs={self.gen_kwargs}, "
            f"use_chat_template={self.use_chat_template}, "
            f"system_prompt={'set' if self.system_prompt else 'none'}"
        )

    def _post_checkpoint_load(self, last_step: int) -> None:
        """Create reference model and reward worker after checkpoint loading."""
        if dist.is_initialized():
            dist.barrier()

        self.logger.info("Creating reference model for GRPO...")
        self.reference_model = self._create_reference_model()

        # Initialize reward worker with config-specific kwargs
        self.logger.info(f"Initializing reward worker (type={self._reward_config.type})...")
        reward_kwargs = {"timeout": self._reward_config.timeout}

        if self._reward_config.type == "api":
            reward_kwargs["provider"] = self._reward_config.api_provider
            if self._reward_config.api_model:
                reward_kwargs["model"] = self._reward_config.api_model
            if self._reward_config.prompt_template:
                reward_kwargs["prompt_template"] = self._reward_config.prompt_template
        elif self._reward_config.type == "local_endpoint":
            reward_kwargs["endpoint"] = self._reward_config.local_endpoint
        elif self._reward_config.type == "local_inference":
            if self._reward_config.local_model_path:
                reward_kwargs["model_path"] = self._reward_config.local_model_path
            reward_kwargs["device"] = self._reward_config.local_device
            reward_kwargs["dtype"] = self._reward_config.local_dtype
            reward_kwargs["load_in_8bit"] = self._reward_config.load_in_8bit
            reward_kwargs["load_in_4bit"] = self._reward_config.load_in_4bit
        elif self._reward_config.type == "format":
            if self._reward_config.required_tags:
                reward_kwargs["required_tags"] = self._reward_config.required_tags
            reward_kwargs["penalty"] = self._reward_config.format_penalty
        elif self._reward_config.type == "keyword":
            reward_kwargs["keyword"] = self._reward_config.keyword
            reward_kwargs["case_sensitive"] = self._reward_config.keyword_case_sensitive
        elif self._reward_config.type == "soft_keyword":
            reward_kwargs["keyword"] = self._reward_config.keyword
            reward_kwargs["case_sensitive"] = self._reward_config.keyword_case_sensitive

        reward_fn = get_reward_function(self._reward_config.type, **reward_kwargs)
        self.reward_worker = RewardWorkerPool(
            reward_fn=reward_fn,
            num_workers=self._reward_config.num_workers,
            timeout=self._reward_config.timeout,
        )

        # Setup GRPO-specific data iterators (overrides base trainer's)
        self._setup_data_iterators()

        if dist.is_initialized():
            dist.barrier()

    def _create_reference_model(self) -> nn.Module:
        """Create frozen reference model from current policy.

        For FSDP, we gather the full state dict and create a non-FSDP reference
        model. Parameters are stored on GPU for faster inference during GRPO training.
        """

        self.logger.info("Creating reference model from policy weights...")
        device = self._get_compute_device()

        # Get the underlying model (handle FSDP wrapping)
        if isinstance(self.model, FSDP):
            from torch.distributed.fsdp import StateDictType
            from ironcore.parallel.parallel import initialize_parallelism

            # For FSDP, we must shard the reference model as well to save memory.
            # 1. Create a raw model instance
            unwrapped = getattr(self.model, "module", self.model)
            reference_model = unwrapped.__class__(unwrapped.config)
            
            # 2. Disable gradients before wrapping (saves memory/compute)
            reference_model.eval()
            for param in reference_model.parameters():
                param.requires_grad = False
                
            # 3. Wrap identically to policy model
            reference_model = initialize_parallelism(self.config, reference_model)
            
            # 4. Copy local sharded state dict directly (no gathering needed)
            with FSDP.state_dict_type(self.model, StateDictType.LOCAL_STATE_DICT):
                local_state_dict = self.model.state_dict()
            with FSDP.state_dict_type(reference_model, StateDictType.LOCAL_STATE_DICT):
                reference_model.load_state_dict(local_state_dict, strict=False)

            self.logger.info(f"Reference model created on GPU (FSDP mode, local sharded copy)")
        else:
            # Handle DDP or unwrapped model - store on GPU for faster inference
            model = getattr(self.model, "module", self.model)
            reference_model = copy.deepcopy(model)
            reference_model = reference_model.to(device)
            reference_model.eval()
            for param in reference_model.parameters():
                param.requires_grad = False

            self.logger.info(f"Reference model created on GPU (device={device})")

        return reference_model

    def _get_ref_log_probs(self, rollout: RolloutBuffer) -> torch.Tensor:
        """Pre-compute reference model log probabilities for generated completions.

        Reference model is kept on GPU for fast inference during GRPO training.
        Memory is explicitly freed after computing log probs.
        """
        if self.reference_model is None:
            raise RuntimeError("Reference model not initialized. Call _post_checkpoint_load first.")

        device = self._get_compute_device()
        total_samples = rollout.total_samples
        
        # Determine micro-batch size for reference inference
        # Match rollout chunk size for memory consistency
        micro_batch_size = rollout.batch_size * (self.group_size // self.rollout_chunks)
        
        all_ref_log_probs = []
        
        for i in range(0, total_samples, micro_batch_size):
            stop = min(i + micro_batch_size, total_samples)
            mb_completion_ids = rollout.completion_ids[i:stop].to(device)
            mb_response_ids = rollout.response_ids[i:stop]
            
            # Prepare labels and mask for this micro-batch
            # We can't use _prepare_labels_and_mask directly as it expects a RolloutBuffer
            prompt_len = rollout.prompt_ids.size(1)
            mb_labels = mb_completion_ids.detach().clone()
            mb_labels[:, :-1] = mb_completion_ids[:, 1:]
            mb_labels[:, -1] = -100
            mb_labels[:, : prompt_len - 1] = -100
            
            mb_response_mask = torch.zeros_like(mb_labels, dtype=torch.float)
            mb_response_mask[:, prompt_len - 1 : -1] = 1.0

            # Reference model is on GPU - direct inference
            ref_output = self.reference_model(mb_completion_ids, labels=None)
            ref_logits = ref_output[0] if isinstance(ref_output, tuple) else ref_output
            
            mb_ref_log_probs = self._compute_token_log_probs_from_logits(
                ref_logits, mb_labels, mb_response_mask
            )
            
            all_ref_log_probs.append(mb_ref_log_probs.detach().cpu())
            
            # Free reference logits immediately
            del ref_logits
            del mb_completion_ids
            torch.cuda.empty_cache()

        return torch.cat(all_ref_log_probs, dim=0).to(device)

    def _prepare_labels_and_mask(self, rollout: RolloutBuffer) -> tuple[torch.Tensor, torch.Tensor]:
        """Centralized logic for label shifting and response masking.

        Uses per-sequence response_lengths when available so that padding tokens
        after EOS are correctly excluded from the loss.
        """
        prompt_len = rollout.prompt_ids.size(1)

        # Create labels (shift by 1 for next-token prediction)
        labels = rollout.completion_ids.clone()
        labels[:, :-1] = rollout.completion_ids[:, 1:]
        labels[:, -1] = -100
        labels[:, : prompt_len - 1] = -100

        # Mask: only compute loss on actual response tokens (EOS-aware)
        response_mask = torch.zeros_like(labels, dtype=torch.float)
        if rollout.response_lengths is not None:
            # Per-sequence mask: [prompt_len-1 : prompt_len-1 + resp_len] for each row
            for i, resp_len in enumerate(rollout.response_lengths.tolist()):
                resp_len = int(resp_len)
                response_mask[i, prompt_len - 1 : prompt_len - 1 + resp_len] = 1.0
        else:
            response_mask[:, prompt_len - 1 : -1] = 1.0

        return labels, response_mask

    def _get_compute_device(self) -> torch.device:
        """Get the device where computation should happen."""
        if isinstance(self.model, FSDP):
            return torch.device(f"cuda:{torch.cuda.current_device()}")
        return next(self.model.parameters()).device

    def _move_batch_to_device(self, batch: dict) -> dict:
        """Move all tensors in batch to model device."""
        device = self._get_compute_device()
        return {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

    def _prepare_prompt_ids(self, prompts: list[str], device: torch.device) -> torch.Tensor:
        """Prepare prompt IDs with optional chat template and system prompt.

        Args:
            prompts: List of raw prompt strings [B]
            device: Target device

        Returns:
            [B, prompt_len] prompt token IDs
        """
        if self.use_chat_template:
            # Build messages and apply chat template
            all_prompt_ids = []
            for prompt in prompts:
                messages = []
                if self.system_prompt:
                    messages.append({"role": "system", "content": self.system_prompt})
                messages.append({"role": "user", "content": prompt})

                prompt_enc = self._tokenizer.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    return_tensors="pt",
                )
                # apply_chat_template with return_tensors="pt" returns BatchEncoding (dict-like)
                prompt_ids = prompt_enc["input_ids"].squeeze(0)
                all_prompt_ids.append(prompt_ids)

            # Pad to same length
            from torch.nn.utils.rnn import pad_sequence

            return pad_sequence(all_prompt_ids, batch_first=True, padding_value=0).to(device)
        else:
            # Fallback: raw tokenization (dataset already tokenized, but re-tokenize if system_prompt)
            if self.system_prompt:
                all_prompt_ids = []
                system_ids = self._tokenizer.encode(self.system_prompt, add_special_tokens=False)

                for prompt in prompts:
                    prompt_ids = self._tokenizer.encode(prompt, add_special_tokens=False)
                    combined = system_ids + prompt_ids
                    all_prompt_ids.append(torch.tensor(combined, dtype=torch.long))

                from torch.nn.utils.rnn import pad_sequence

                return pad_sequence(all_prompt_ids, batch_first=True, padding_value=0).to(device)
            else:
                # Use pre-tokenized IDs from batch (handled by caller)
                return None

    def _setup_data_iterators(self) -> None:
        """Setup data iterators for training and evaluation."""
        self.data_iterator = {
            "train": get_grpo_data_iterator(self.config, split="train"),
        }

        if hasattr(self.config.data, "eval_file") and self.config.data.eval_file:
            self.data_iterator["eval"] = get_grpo_data_iterator(self.config, split="eval")

    def _forward_micro_batch(self, step: int) -> tuple[torch.Tensor, dict[str, float] | None]:
        """Forward pass for a single micro-batch in GRPO training."""
        raise NotImplementedError("GRPO uses custom train_step, not _forward_micro_batch")

    def train_step(self, step: int) -> tuple[float, float, float]:
        """GRPO training step.

        Phase 1 — Rollout (runs once per step):
          1. Sample prompts from dataset
          2. Generate G completions per prompt (batched prefix KV-cache)
          3. Compute rewards via worker pool
          4. Compute group-wise advantages
          old_log_probs are frozen from generation time.

        Phase 2 — Update (runs grpo_num_epochs times):
          - Online  (num_epochs=1): standard policy gradient, ratio=1
          - Offline (num_epochs>1): IS ratio = π_θ / π_old, optionally PPO-clipped
        """
        self.timer.start(name="iter")

        # === Phase 1: Rollout (once) ===
        batch = next(self.data_iterator["train"])
        batch = self._move_batch_to_device(batch)

        prompts = batch["prompts"]  # Raw text prompts
        metadata = batch["metadata"]
        G = self.group_size

        # Prepare prompt IDs (with optional chat template / system prompt)
        device = self._get_compute_device()
        if self.use_chat_template or self.system_prompt:
            prompt_ids = self._prepare_prompt_ids(prompts, device)
        else:
            prompt_ids = batch["input_ids"]

        self.model.eval()
        with torch.no_grad():
            # Chunked rollout generation for larger effective group sizes
            # When rollout_chunks > 1, generate group_size/rollout_chunks per chunk
            # This keeps memory constant while allowing larger total group sizes
            rollout = None
            chunk_group_size = G // self.rollout_chunks
            chunk_metadata = metadata  # Same metadata for each chunk (will be expanded)

            for chunk_idx in range(self.rollout_chunks):
                chunk_rollout = generate_rollouts_batched(
                    model=self.model,
                    prompt_ids=prompt_ids,
                    group_size=chunk_group_size,
                    metadata=chunk_metadata,
                    eos_token_id=self._tokenizer.eos_token_id,
                    **self.gen_kwargs,
                )
                if rollout is None:
                    rollout = chunk_rollout
                else:
                    rollout = rollout.cat(chunk_rollout)

        # Explicitly clear generation cache before starting training phase
        torch.cuda.empty_cache()

        self.model.train()

        # Safety check - rollout should never be None here
        if rollout is None:
            raise RuntimeError("Rollout generation failed - no chunks were generated")

        if self.reward_worker is None:
            raise RuntimeError("Reward worker not initialized. Call _post_checkpoint_load first.")

        prompts_text = self._tokenizer.batch_decode(prompt_ids, skip_special_tokens=True)
        responses_text = self._tokenizer.batch_decode(
            rollout.response_ids, skip_special_tokens=True
        )

        # Store for metrics computation
        self._current_responses_text = responses_text
        # Compute response lengths (non-padding tokens)
        self._current_response_lengths = (
            (rollout.response_ids != self._tokenizer.pad_token_id).sum(dim=1).float()
        )

        # Match chunk-interleaved response layout:
        # cat() output: [chunk0_B0*, chunk0_B1*, ..., chunk1_B0*, chunk1_B1*, ...]
        # repeated_prompts must follow the same order.
        chunk_group_size = G // self.rollout_chunks
        repeated_prompts = [
            p
            for _ in range(self.rollout_chunks)
            for p in prompts_text
            for _ in range(chunk_group_size)
        ]
        rewards = self.reward_worker.score_batch(
            prompts=repeated_prompts,
            completions=responses_text,
            metadata_list=rollout.metadata,
        )
        rewards = rewards.to(prompt_ids.device)

        advantages = compute_advantages(rewards, rollout.group_ids, self.eps, distributed=False)

        # old_log_probs frozen at generation time — used for IS in offline epochs
        old_log_probs = rollout.old_log_probs.detach()

        # Pre-compute reference log probs once per rollout (static across epochs)
        with torch.no_grad():
            ref_log_probs = self._get_ref_log_probs(rollout)

        # === Phase 2: Multi-epoch update ===
        metrics: dict[str, float] = {}
        grad_norm = param_norm = 0.0
        
        # Determine micro-batch size for update phase
        # We want to process at most B * (G / chunks) at once to match rollout memory
        micro_batch_size = prompt_ids.size(0) * (G // self.rollout_chunks)
        total_samples = rollout.total_samples

        for epoch in range(self.num_epochs):
            # Shuffle indices within the rollout to ensure varied micro-batches
            perm = torch.randperm(total_samples, device=device)
            
            for i in range(0, total_samples, micro_batch_size):
                stop = min(i + micro_batch_size, total_samples)
                mb_indices = perm[i:stop]
                
                # Create a temporary micro-batch buffer
                # Note: We need a lightweight way to slice the rollout
                mb_rollout = rollout.select(mb_indices)
                mb_advantages = advantages[mb_indices]
                mb_ref_log_probs = ref_log_probs[mb_indices]
                mb_old_log_probs = old_log_probs[mb_indices] if self.num_epochs > 1 else None

                loss, mb_metrics = self._compute_grpo_loss(
                    mb_rollout,
                    mb_advantages,
                    mb_ref_log_probs,
                    old_log_probs=mb_old_log_probs,
                )
                
                # Scale loss by micro-batch fraction (accumulation is defined by rollout batch size)
                scaled_loss = loss * (len(mb_indices) / total_samples)
                self.scaler.scale(scaled_loss).backward()
                
                # Accumulate metrics
                for k, v in mb_metrics.items():
                    metrics[k] = metrics.get(k, 0.0) + v * (len(mb_indices) / total_samples)

            grad_norm, param_norm = self._compute_grad_and_param_norms(step)
            self._optimizer_step()

        self.timer.stop(name="iter")

        self._check_loss_for_nan(metrics["grpo_loss"], step)

        if is_first_rank() and self.control.do_log(step):
            self._log_grpo_metrics(step, metrics, rewards, advantages)

            # Qualitative sample logging at metrics_interval
            metrics_interval = self.config.alignment.metrics_interval
            if metrics_interval > 0 and step % metrics_interval == 0:
                self._log_qualitative_samples(
                    prompts_text, responses_text, rollout.metadata, rewards, rollout.group_ids, step
                )

        return metrics["grpo_loss"], grad_norm, param_norm

    def _compute_grpo_loss(
        self,
        rollout: RolloutBuffer,
        advantages: torch.Tensor,
        ref_log_probs: torch.Tensor,
        old_log_probs: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute GRPO loss with current policy."""
        device = self._get_compute_device()
        labels, response_mask = self._prepare_labels_and_mask(rollout)

        # Compute policy log probs for generated tokens
        policy_output = self.model(rollout.completion_ids.to(device), labels=None)
        # Handle tuple return (logits, kv_cache) when model is in eval mode
        policy_logits = policy_output[0] if isinstance(policy_output, tuple) else policy_output
        policy_log_probs_token = self._compute_token_log_probs_from_logits(
            policy_logits, labels, response_mask
        )

        # Compute approximate KL divergence using token-level log probs
        # kl_divergence_approx: (policy_log_probs - ref_log_probs).sum(dim=-1)
        kl_per_seq = kl_divergence_approx(
            policy_log_probs_token, ref_log_probs.to(device), response_mask
        )

        # Sum token-level log probs for sequence-level advantage multiplication
        policy_log_probs_seq = (policy_log_probs_token * response_mask).sum(dim=-1)
        ref_log_probs_seq = (ref_log_probs.to(device) * response_mask).sum(dim=-1)

        # GRPO loss — pass old_log_probs for IS when doing offline multi-epoch updates
        loss, metrics = grpo_loss(
            policy_log_probs=policy_log_probs_seq,
            ref_log_probs=ref_log_probs_seq,
            advantages=advantages.to(device),
            kl_per_seq=kl_per_seq,
            beta=self.beta,
            old_log_probs=old_log_probs.to(device) if old_log_probs is not None else None,
            clip_eps=self.clip_eps,
        )

        return loss, metrics

    def _compute_token_log_probs_from_logits(
        self, logits: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        """Compute per-token log probs from logits (TP-safe, chunked for memory).

        Returns: [batch, seq_len] log probabilities for tokens in labels.
        """
        batch_size, seq_len, vocab_size = logits.shape

        # Output tensor
        per_token_logps = torch.zeros(batch_size, seq_len, dtype=torch.float, device=logits.device)

        # Process in chunks along sequence dimension to reduce peak memory
        # 128-256 is a good balance between memory and kernel efficiency
        chunk_size = min(128, seq_len)

        for start in range(0, seq_len, chunk_size):
            end = min(start + chunk_size, seq_len)
            logits_chunk = logits[:, start:end, :]  # [batch, chunk, vocab]
            labels_chunk = labels[:, start:end]  # [batch, chunk]

            # Compute log_softmax only for this chunk
            log_probs_chunk = _compute_log_softmax_tp_safe(logits_chunk)

            # Extract log probs for actual tokens in chunk
            valid_mask = labels_chunk != -100
            if valid_mask.any():
                indices = labels_chunk[valid_mask].unsqueeze(-1)
                token_logps = log_probs_chunk[valid_mask].gather(dim=-1, index=indices).squeeze(-1)
                per_token_logps[:, start:end][valid_mask] = token_logps

        return per_token_logps

    def _compute_sequence_log_probs_from_logits(
        self, logits: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        """Compute sequence log probs from logits (TP-safe)."""
        log_probs = _compute_log_softmax_tp_safe(logits)
        return _extract_logps_from_log_probs(log_probs, labels, mask)

    def _compute_grad_and_param_norms(self, step: int) -> tuple[float, float]:
        """Compute gradient and parameter norms."""
        grad_norm = 0.0
        param_norm = 0.0

        # Unwrap model for norm computation
        model = self.model.module if hasattr(self.model, "module") else self.model

        for p in model.parameters():
            if p.grad is not None:
                grad_norm += p.grad.data.norm(2).item() ** 2
            param_norm += p.data.norm(2).item() ** 2

        grad_norm = grad_norm**0.5
        param_norm = param_norm**0.5

        return grad_norm, param_norm

    def _optimizer_step(self) -> None:
        """Perform optimizer step with gradient scaling."""
        self.scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(
            self.model.parameters(),
            self.config.optim.clip_grad,
        )
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad()
        self.lr_scheduler.step()

    def _log_grpo_metrics(
        self, step: int, metrics: dict, rewards: torch.Tensor, advantages: torch.Tensor
    ) -> None:
        """Log GRPO-specific metrics."""
        for name, value in metrics.items():
            log_metric(f"grpo/{name}", value, step)

        log_metric("grpo/mean_reward", rewards.mean().item(), step)
        log_metric("grpo/std_reward", rewards.std().item() if len(rewards) > 1 else 0.0, step)
        log_metric("grpo/mean_advantage", advantages.mean().item(), step)

        # Compute response length metrics
        # rollout.response_ids has shape [B*G, response_len]
        # Get mean response length across all completions
        response_lengths = self._current_response_lengths
        if response_lengths is not None:
            log_metric("grpo/mean_response_length", response_lengths.mean().item(), step)

        # Compute format compliance (#### in response)
        responses_text = self._current_responses_text
        if responses_text:
            format_hits = sum(1 for r in responses_text if "####" in r) / len(responses_text)
            log_metric("grpo/keyword_hit_rate", format_hits, step)

        self.logger.info(
            f"step: {step}, grpo_loss: {metrics['grpo_loss']:.4f}, "
            f"policy_loss: {metrics['policy_loss']:.4f}, "
            f"kl_loss: {metrics['kl_loss']:.4f}, "
            f"mean_reward: {rewards.mean().item():.4f}, "
            f"mean_ratio: {metrics['mean_ratio']:.4f}, "
            f"clip_frac: {metrics['clip_fraction']:.3f}"
        )

    def _log_qualitative_samples(
        self,
        prompts_text: list[str],
        responses_text: list[str],
        metadata: list[dict],
        rewards: torch.Tensor,
        group_ids: torch.Tensor,
        step: int,
    ) -> None:
        """Log qualitative samples for debugging.

        Logs the first prompt and its best/worst completions.
        Uses group_ids for correct indexing regardless of rollout_chunks layout.
        """
        num_prompts = len(prompts_text)

        # Log up to 3 prompts
        for i in range(min(3, num_prompts)):
            # Use group_ids to find all completions belonging to prompt i
            group_indices = (group_ids == i).nonzero(as_tuple=True)[0]
            group_responses = [responses_text[idx] for idx in group_indices.tolist()]
            group_rewards = rewards[group_indices]

            # Find best and worst in group
            best_idx = int(group_rewards.argmax().item())
            worst_idx = int(group_rewards.argmin().item())

            prompt_preview = (
                prompts_text[i][:100] + "..." if len(prompts_text[i]) > 100 else prompts_text[i]
            )
            best_response = group_responses[best_idx]
            worst_response = group_responses[worst_idx]
            ground_truth = metadata[int(group_indices[0].item())].get("answer", "N/A")

            self.logger.info(
                f"\n{'=' * 60}\n"
                f"[Step {step}] Sample {i}\n"
                f"{'=' * 60}\n"
                f"Prompt: {prompt_preview}\n"
                f"Ground truth: {ground_truth}\n"
                f"--- Best (reward={group_rewards[best_idx].item():.2f}) ---\n"
                f"{best_response[:500]}{'...' if len(best_response) > 500 else ''}\n"
                f"--- Worst (reward={group_rewards[worst_idx].item():.2f}) ---\n"
                f"{worst_response[:300]}{'...' if len(worst_response) > 300 else ''}\n"
                f"{'=' * 60}"
            )

    def _eval_step(self, data_iterator: Iterator) -> tuple[float, float]:
        """Evaluation step (simplified - just compute loss on held-out prompts)."""
        # For GRPO, evaluation is tricky since we need to generate + reward
        # For now, return placeholder
        return 0.0, 0.0
