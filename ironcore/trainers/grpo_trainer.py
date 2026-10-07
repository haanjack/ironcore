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
from contextlib import nullcontext

import torch
from torch import distributed as dist
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

from ironcore import get_tokenizer
from ironcore.alignment.buffer import RolloutBuffer
from ironcore.alignment.dataset import get_grpo_data_iterator
from ironcore.alignment.loss.dpo import _compute_log_softmax_tp_safe, _extract_logps_from_log_probs
from ironcore.alignment.loss.grpo import compute_advantages, compute_entropy, grpo_loss
from ironcore.alignment.loss.kl import kl_divergence_approx
from ironcore.alignment.rewards import RewardManager, RewardWorkerPool
from ironcore.alignment.rollout import generate_rollouts_batched
from ironcore.global_vars import log_metric
from ironcore.parallel import parallel_states
from ironcore.utils import is_first_rank

from .base_trainer import BaseTrainer


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
        self.entropy_coef = getattr(config.alignment, "grpo_entropy_coef", 0.0)
        self.rollout_micro_group_size = config.alignment.grpo_rollout_micro_group_size
        self.rollout_chunks = self.group_size // self.rollout_micro_group_size
        self.use_paged_rollout = getattr(config.alignment, "grpo_use_paged_rollout", False)

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

        if is_first_rank() and torch.cuda.is_available():
            dev = self._get_compute_device()
            self.logger.info(
                f"[VRAM] After ref model creation: "
                f"allocated={torch.cuda.memory_allocated(dev) / 1024**3:.2f} GB, "
                f"reserved={torch.cuda.memory_reserved(dev) / 1024**3:.2f} GB"
            )

        # Offload reference model to CPU to free GPU memory for policy model
        offload_ref = getattr(self.config.alignment, "offload_ref_model", False)
        is_fsdp = isinstance(self.model, FSDP)
        if offload_ref:
            if is_fsdp:
                self.logger.warning(
                    "offload_ref_model=True is not compatible with FSDP. "
                    "Reference model will remain on GPU. Disable FSDP to enable offloading."
                )
            else:
                self.reference_model = self.reference_model.to("cpu")
                self.logger.info("Reference model offloaded to CPU")

        # Initialize reward worker via RewardManager
        reward_manager_cfg = self.config.alignment.reward_manager
        # BaseConfig.__call__ cannot convert Union[X, None] typed fields, so the
        # reward_manager may arrive as a raw dict from the YAML loader. Normalize here.
        if isinstance(reward_manager_cfg, dict):
            from ironcore.config.config_alignment import RewardManagerConfig

            reward_manager_cfg = RewardManagerConfig(**reward_manager_cfg)

        if reward_manager_cfg is None:
            raise ValueError("GRPO requires reward_manager configuration")

        self.logger.info("Initializing reward worker via RewardManager...")
        reward_fn = RewardManager.from_config(reward_manager_cfg)
        self.reward_worker = RewardWorkerPool(
            reward_fn=reward_fn,
            num_workers=reward_manager_cfg.num_workers,
            timeout=reward_manager_cfg.timeout,
            failure_policy=reward_manager_cfg.failure_policy,
        )

        # Data iterators are already set up via _get_data_iterator() override

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

            # For FSDP, we must shard the reference model as well to save memory.
            # 1. Create a raw model instance
            # Use self.config.model directly as FSDP does not proxy .config
            from ironcore.language_model import LanguageModel
            from ironcore.parallel.parallel import initialize_parallelism

            reference_model = LanguageModel(self.config)
            reference_model.to(device=device, dtype=next(self.model.parameters()).dtype)

            # 2. Cast to match policy model dtype before FSDP wrapping.
            # LanguageModel initializes in fp32 by default; the policy model loads
            # HF weights that are typically bf16. FSDP mixed_precision casts
            # parameters but not activation tensors, so FlashAttention sees fp32
            # q/k/v if we don't cast here.
            fsdp_mp = getattr(self.config.parallel, "fsdp_mixed_precision", "none")
            del fsdp_mp  # compute casts belong to FSDP/autocast, not stored weights

            # 3. Disable gradients before wrapping (saves memory/compute)
            reference_model.eval()
            for param in reference_model.parameters():
                param.requires_grad = False

            # 4. Wrap identically to policy model
            reference_model = initialize_parallelism(self.config, reference_model)

            # 5. Copy local sharded state dict directly (no gathering needed)
            with FSDP.state_dict_type(self.model, StateDictType.LOCAL_STATE_DICT):
                local_state_dict = self.model.state_dict()
            with FSDP.state_dict_type(reference_model, StateDictType.LOCAL_STATE_DICT):
                reference_model.load_state_dict(local_state_dict, strict=False)

            self.logger.info("Reference model created on GPU (FSDP mode, local sharded copy)")
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
        When offload_ref_model is True, the model is moved to GPU for the inference
        loop and back to CPU afterward.
        Memory is explicitly freed after computing log probs.
        """
        if self.reference_model is None:
            raise RuntimeError("Reference model not initialized. Call _post_checkpoint_load first.")

        device = self._get_compute_device()
        total_samples = rollout.total_samples

        # Performance optimization: Accumulate on GPU by default to avoid sync overhead.
        # If memory is an issue, this can be moved back to CPU via config.
        offload_to_cpu = getattr(self.config.alignment, "grpo_offload_ref_logps", False)

        # Move reference model to GPU if it was offloaded to CPU
        offload_ref_model = getattr(self.config.alignment, "offload_ref_model", False)
        ref_on_cpu = offload_ref_model and not isinstance(self.model, FSDP)
        if ref_on_cpu:
            self.reference_model = self.reference_model.to(device)

        # Determine micro-batch size for reference inference
        # Match rollout chunk size for memory consistency
        micro_batch_size = rollout.batch_size * self.rollout_micro_group_size

        all_ref_log_probs = []

        for i in range(0, total_samples, micro_batch_size):
            stop = min(i + micro_batch_size, total_samples)
            mb_completion_ids = rollout.completion_ids[i:stop].to(device)

            # Use select() to get a proper sub-buffer, then reuse _prepare_labels_and_mask
            # for consistent EOS-aware masking between policy and reference
            mb_indices = torch.arange(i, stop, device=device)
            mb_rollout = rollout.select(mb_indices)
            mb_labels, mb_response_mask = self._prepare_labels_and_mask(mb_rollout)

            # Reference model is on GPU - direct inference
            ref_output = self.reference_model(
                mb_completion_ids, labels=None, **self._rollout_forward_kwargs(mb_rollout)
            )
            ref_logits = ref_output[0] if isinstance(ref_output, tuple) else ref_output
            from ironcore.training_utils import clear_moe_aux_loss

            clear_moe_aux_loss(self.reference_model)

            mb_ref_log_probs = self._compute_token_log_probs_from_logits(
                ref_logits, mb_labels, mb_response_mask
            )

            if offload_to_cpu:
                all_ref_log_probs.append(mb_ref_log_probs.detach().cpu())
            else:
                all_ref_log_probs.append(mb_ref_log_probs.detach())

            # Free reference logits immediately; allocator will manage reuse
            del ref_logits
            del mb_completion_ids

        res = torch.cat(all_ref_log_probs, dim=0)

        # Move reference model back to CPU if offloading is enabled
        if ref_on_cpu:
            self.reference_model = self.reference_model.to("cpu")

        return res.to(device) if offload_to_cpu else res

    def _rollout_forward_kwargs(self, rollout):
        prompt_mask = rollout.prompt_attention_mask.to(rollout.completion_ids.device).bool()
        if bool(prompt_mask.all()):
            return {}
        local_prompt_ids = rollout.group_ids % rollout.batch_size
        mask = prompt_mask[local_prompt_ids]
        response_width = rollout.completion_ids.size(1) - mask.size(1)
        keys = torch.cat(
            [mask, torch.ones(mask.size(0), response_width, device=mask.device, dtype=torch.bool)],
            dim=1,
        )
        return {
            "position_ids": (keys.long().cumsum(-1) - 1).clamp(min=0),
            "attention_mask": keys[:, None, None, :].expand(-1, 1, keys.size(1), -1),
        }

    def _left_pad_prompts(self, ids, mask):
        lengths = mask.sum(-1)
        if (lengths == 0).any():
            raise ValueError("GRPO prompts must contain at least one valid token")
        width = int(lengths.max())
        output = ids.new_full((ids.size(0), width), self._tokenizer.pad_token_id or 0)
        valid = torch.zeros_like(output, dtype=torch.bool)
        for i, length in enumerate(lengths.tolist()):
            output[i, -length:] = ids[i][mask[i].bool()]
            valid[i, -length:] = True
        return output, valid

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
                    return_dict=True,
                )
                # apply_chat_template with return_tensors="pt" returns BatchEncoding (dict-like)
                prompt_ids = prompt_enc["input_ids"].squeeze(0)
                all_prompt_ids.append(prompt_ids)

            # Pad to same length
            from torch.nn.utils.rnn import pad_sequence

            self._prepared_prompt_mask = pad_sequence(
                [torch.ones_like(ids, dtype=torch.bool) for ids in all_prompt_ids],
                batch_first=True,
                padding_value=False,
                padding_side="left",
            ).to(device)
            return pad_sequence(
                all_prompt_ids,
                batch_first=True,
                padding_value=self._tokenizer.pad_token_id or 0,
                padding_side="left",
            ).to(device)
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

                self._prepared_prompt_mask = pad_sequence(
                    [torch.ones_like(ids, dtype=torch.bool) for ids in all_prompt_ids],
                    batch_first=True,
                    padding_value=False,
                    padding_side="left",
                ).to(device)
                return pad_sequence(
                    all_prompt_ids,
                    batch_first=True,
                    padding_value=self._tokenizer.pad_token_id or 0,
                    padding_side="left",
                ).to(device)
            else:
                # Use pre-tokenized IDs from batch (handled by caller)
                return None

    def _get_data_iterator(self):
        """Return GRPO-specific data iterators."""
        iterators = {
            "train": get_grpo_data_iterator(self.config, split="train"),
        }
        if self.config.data.eval_datasets:
            iterators["eval"] = get_grpo_data_iterator(self.config, split="eval")
        return iterators

    def _skip_consumed_train_samples(self, last_step: int) -> None:
        """Each GRPO step consumes one local rollout prompt batch."""
        if hasattr(self.data_iterator.get("train"), "load_state_dict"):
            return super()._skip_consumed_train_samples(last_step)
        if "train" in self.data_iterator:
            for _ in range(max(0, last_step)):
                next(self.data_iterator["train"])

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
        device = self._get_compute_device()

        if is_first_rank() and torch.cuda.is_available() and step == 0:
            torch.cuda.reset_peak_memory_stats(device)

        # === Phase 1: Rollout (once) ===
        batch = next(self.data_iterator["train"])
        batch = self._move_batch_to_device(batch)

        prompts = batch["prompts"]  # Raw text prompts
        metadata = batch["metadata"]

        # Prepare prompt IDs (with optional chat template / system prompt)
        if self.use_chat_template or self.system_prompt:
            prompt_ids = self._prepare_prompt_ids(prompts, device)
        else:
            prompt_ids = batch["input_ids"]

        raw_mask = batch.get("attention_mask", torch.ones_like(prompt_ids))
        if self.use_chat_template or self.system_prompt:
            raw_mask = self._prepared_prompt_mask
        prompt_ids, prompt_mask = self._left_pad_prompts(prompt_ids, raw_mask)
        if self.use_paged_rollout and not bool(prompt_mask.all()):
            raise ValueError(
                "Padded GRPO prompts currently require batched rollout (grpo_use_paged_rollout=False)"
            )
        self.model.eval()
        with torch.no_grad(), self.context["autocast"]:
            # Chunked rollout: generate rollout_micro_group_size completions per chunk
            # chunks = group_size / micro_group_size (derived at init)
            rollout = None
            chunk_group_size = self.rollout_micro_group_size
            chunk_metadata = metadata  # Same metadata for each chunk (will be expanded)

            if self.use_paged_rollout:
                from ironcore.alignment.rollout import generate_rollouts_paged

                unwrapped = getattr(self.model, "module", self.model)
                unwrapped.initialize_cache(
                    prompt_ids.size(0) + prompt_ids.size(0) * chunk_group_size,
                    prompt_ids.device,
                )

            for chunk_idx in range(self.rollout_chunks):
                if self.use_paged_rollout:
                    chunk_rollout = generate_rollouts_paged(
                        model=self.model,
                        prompt_ids=prompt_ids,
                        group_size=chunk_group_size,
                        metadata=chunk_metadata,
                        eos_token_id=self._tokenizer.eos_token_id,
                        **self.gen_kwargs,
                    )
                else:
                    chunk_rollout = generate_rollouts_batched(
                        model=self.model,
                        prompt_ids=prompt_ids,
                        prompt_attention_mask=prompt_mask,
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

        if is_first_rank() and torch.cuda.is_available():
            peak_after_rollout = torch.cuda.max_memory_allocated(device)
            self.logger.info(f"[VRAM] After rollout: peak={peak_after_rollout / 1024**3:.2f} GB")

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
            rollout.response_lengths.float()
            if rollout.response_lengths is not None
            else (rollout.response_ids != self._tokenizer.pad_token_id).sum(dim=1).float()
        )

        # Match chunk-interleaved response layout:
        # cat() output: [chunk0_B0*, chunk0_B1*, ..., chunk1_B0*, chunk1_B1*, ...]
        # repeated_prompts must follow the same order.
        repeated_prompts = [
            p
            for _ in range(self.rollout_chunks)
            for p in prompts_text
            for _ in range(self.rollout_micro_group_size)
        ]
        rewards = self.reward_worker.score_batch(
            prompts=repeated_prompts,
            completions=responses_text,
            metadata_list=rollout.metadata,
        )
        rewards = rewards.to(prompt_ids.device)

        self._validate_distributed_loss(float(rewards.mean()), step)
        advantages = compute_advantages(rewards, rollout.group_ids, self.eps, distributed=True)

        # old_log_probs frozen at generation time — used for IS in offline epochs
        old_log_probs = rollout.old_log_probs.detach()

        # Pre-compute reference log probs once per rollout (static across epochs)
        with torch.no_grad(), self.context["autocast"]:
            ref_log_probs = self._get_ref_log_probs(rollout)

        if is_first_rank() and torch.cuda.is_available():
            peak_after_ref = torch.cuda.max_memory_allocated(device)
            self.logger.info(f"[VRAM] After ref log probs: peak={peak_after_ref / 1024**3:.2f} GB")

        # === Phase 2: Multi-epoch update ===
        metrics: dict[str, float] = {}
        grad_norm = param_norm = 0.0

        # Determine micro-batch size for update phase
        # Process B_local * micro_group_size samples at once (matches rollout memory)
        micro_batch_size = prompt_ids.size(0) * self.rollout_micro_group_size
        total_samples = rollout.total_samples

        for epoch in range(self.num_epochs):
            # Shuffle indices within the rollout to ensure varied micro-batches
            perm = torch.randperm(total_samples, device=device)

            for i in range(0, total_samples, micro_batch_size):
                stop = min(i + micro_batch_size, total_samples)
                is_last_micro_batch = stop == total_samples
                mb_indices = perm[i:stop]

                # Create a temporary micro-batch buffer
                # Note: We need a lightweight way to slice the rollout
                mb_rollout = rollout.select(mb_indices)
                mb_advantages = advantages[mb_indices]
                mb_ref_log_probs = ref_log_probs[mb_indices]
                mb_old_log_probs = old_log_probs[mb_indices] if self.num_epochs > 1 else None

                # DDP requires no_sync to enclose the forward as well as backward.
                backward_sync_ctx = (
                    self.model.no_sync
                    if not is_last_micro_batch and hasattr(self.model, "no_sync")
                    else nullcontext
                )
                with backward_sync_ctx(), self.context["autocast"]:
                    loss, mb_metrics = self._compute_grpo_loss(
                        mb_rollout,
                        mb_advantages,
                        mb_ref_log_probs,
                        old_log_probs=mb_old_log_probs,
                    )
                    scaled_loss = loss * (len(mb_indices) / total_samples)
                    self.scaler.scale(scaled_loss).backward()

                # Accumulate metrics (average over epochs and samples)
                for k, v in mb_metrics.items():
                    weight = len(mb_indices) / (total_samples * self.num_epochs)
                    metrics[k] = metrics.get(k, 0.0) + v * weight

            self._validate_distributed_loss(metrics["grpo_loss"], step)
            grad_norm, param_norm = self._compute_grad_and_param_norms(step)
            self._optimizer_step()

        self.timer.stop(name="iter")
        metrics = self._average_dp_metrics(metrics)

        if is_first_rank() and torch.cuda.is_available():
            peak_after_step = torch.cuda.max_memory_allocated(device)
            self.logger.info(
                f"[VRAM] After optimizer step: peak={peak_after_step / 1024**3:.2f} GB"
            )

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
        policy_output = self.model(
            rollout.completion_ids.to(device), labels=None, **self._rollout_forward_kwargs(rollout)
        )
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
        from ironcore.alignment.rollout import _filter_logits

        generation = self.config.alignment.generation
        if generation.temperature != 1.0 or generation.top_p < 1.0 or generation.top_k > 0:
            filtered_logits = _filter_logits(
                policy_logits.reshape(-1, policy_logits.size(-1)),
                generation.temperature,
                generation.top_p,
                generation.top_k,
            ).view_as(policy_logits)
            policy_log_probs_token = self._compute_token_log_probs_from_logits(
                filtered_logits, labels, response_mask
            )
        # With no warping, policy and KL use the same log-probability graph.
        # Separate softmax graphs round their gradients to BF16 independently
        # before adding them at the logits, rather than summing in FP32 first.
        prompt_length = rollout.prompt_ids.size(1)
        token_old = torch.zeros_like(policy_log_probs_token)
        if rollout.old_token_log_probs is not None:
            token_old[
                :, prompt_length - 1 : prompt_length - 1 + rollout.old_token_log_probs.size(1)
            ] = rollout.old_token_log_probs.to(device)
        elif self.config.alignment.grpo_objective == "grpo" and old_log_probs is not None:
            raise ValueError(
                "Off-policy token GRPO cannot use a legacy sequence-only rollout buffer"
            )

        # Note: We use the sum of log-probabilities for the sequence-level objective.
        # This follows the standard policy gradient formulation for whole-sequence rollouts.
        policy_log_probs_seq = (policy_log_probs_token * response_mask).sum(dim=-1)
        ref_log_probs_seq = (ref_log_probs.to(device) * response_mask).sum(dim=-1)

        # Compute entropy for exploration bonus
        entropy = None
        if self.entropy_coef > 0.0:
            # Compute entropy from logits: H = -sum(p * log p)
            # First convert logits to log_probs via log_softmax
            policy_log_probs_full = _compute_log_softmax_tp_safe(
                policy_logits, logits_are_parallel=False
            )
            entropy = compute_entropy(policy_log_probs_full, response_mask)  # [batch]

        # GRPO loss — pass old_log_probs for IS when doing offline multi-epoch updates.
        # response_lengths enables length normalisation of the IS ratio and KL
        # (Fable issues #68 / #69): longer completions no longer mechanically
        # dominate the gradient or inflate the clip fraction.
        response_lengths = response_mask.sum(dim=-1).clamp(min=1)
        loss, metrics = grpo_loss(
            policy_log_probs=policy_log_probs_seq,
            ref_log_probs=ref_log_probs_seq,
            advantages=advantages.to(device),
            kl_per_seq=kl_per_seq,
            beta=self.beta,
            old_log_probs=old_log_probs.to(device) if old_log_probs is not None else None,
            clip_eps=self.clip_eps,
            entropy=entropy,
            entropy_coef=self.entropy_coef,
            response_lengths=response_lengths,
            objective=self.config.alignment.grpo_objective,
            token_policy_log_probs=policy_log_probs_token,
            token_old_log_probs=token_old,
            response_mask=response_mask,
        )

        from ironcore.training_utils import clear_moe_aux_loss, get_moe_aux_loss

        aux = get_moe_aux_loss(self.model)
        if aux is not None and self.config.alignment.moe_aux_loss == "include":
            loss = loss + aux
            metrics["moe_aux_loss"] = aux.detach()
        clear_moe_aux_loss(self.model)
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
            log_probs_chunk = _compute_log_softmax_tp_safe(logits_chunk, logits_are_parallel=False)

            # Extract log probs for actual tokens in chunk
            valid_mask = (labels_chunk != -100) & mask[:, start:end].bool()
            if valid_mask.any():
                indices = labels_chunk[valid_mask].unsqueeze(-1)
                token_logps = log_probs_chunk[valid_mask].gather(dim=-1, index=indices).squeeze(-1)
                per_token_logps[:, start:end][valid_mask] = token_logps

        return per_token_logps

    def _compute_sequence_log_probs_from_logits(
        self, logits: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        """Compute sequence log probs from logits (TP-safe)."""
        log_probs = _compute_log_softmax_tp_safe(logits, logits_are_parallel=False)
        return _extract_logps_from_log_probs(log_probs, labels, mask)

    def _compute_grad_and_param_norms(self, step: int) -> tuple[float, float]:
        return super()._compute_grad_and_param_norms(step)

    def _optimizer_step(self) -> None:
        """Perform optimizer step with gradient scaling."""
        super()._optimizer_step()

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

    def _eval_step(self, batch) -> tuple[float, float]:
        """GRPO evaluates generated rewards through evaluate()."""
        raise NotImplementedError("GRPO evaluation reports rewards, not language-model loss")

    def evaluate(self, global_step: int):
        """Generate held-out completions and report sample-weighted rewards."""
        if "eval" not in self.data_iterator:
            raise ValueError("GRPO evaluation requires an eval prompt iterator")
        if self.reward_worker is None:
            raise RuntimeError("Reward worker is not initialized")
        device = self._get_compute_device()
        was_training = self.model.training
        self.model.eval()
        totals = torch.zeros(3, device=device, dtype=torch.float64)
        batch_count = max(
            1,
            self.config.operation.eval_samples
            // (
                self.config.trainer.eval_batch_size * parallel_states.get_data_parallel_world_size()
            ),
        )
        try:
            with torch.no_grad(), self.context["autocast"]:
                for batch in self._evaluation_batches(batch_count):
                    batch = self._move_batch_to_device(batch)
                    prompt_ids = (
                        self._prepare_prompt_ids(batch["prompts"], device)
                        if self.use_chat_template or self.system_prompt
                        else batch["input_ids"]
                    )
                    raw_mask = batch.get("attention_mask", torch.ones_like(prompt_ids))
                    if self.use_chat_template or self.system_prompt:
                        raw_mask = self._prepared_prompt_mask
                    prompt_ids, prompt_mask = self._left_pad_prompts(prompt_ids, raw_mask)
                    prompts = self._tokenizer.batch_decode(prompt_ids, skip_special_tokens=True)
                    for _ in range(self.rollout_chunks):
                        rollout = generate_rollouts_batched(
                            model=self.model,
                            prompt_ids=prompt_ids,
                            prompt_attention_mask=prompt_mask,
                            group_size=self.rollout_micro_group_size,
                            metadata=batch["metadata"],
                            eos_token_id=self._tokenizer.eos_token_id,
                            **self.gen_kwargs,
                        )
                        rewards = self.reward_worker.score_batch(
                            prompts=[
                                p for p in prompts for _ in range(self.rollout_micro_group_size)
                            ],
                            completions=self._tokenizer.batch_decode(
                                rollout.response_ids, skip_special_tokens=True
                            ),
                            metadata_list=rollout.metadata,
                        ).to(device)
                        totals += torch.stack(
                            [
                                rewards.sum(),
                                (rewards > 0).sum(),
                                rewards.new_tensor(rewards.numel()),
                            ]
                        )
            from ironcore.parallel.parallel_states import (
                get_data_parallel_group,
                get_data_parallel_world_size,
            )

            if dist.is_initialized() and get_data_parallel_world_size() > 1:
                dist.all_reduce(totals, group=get_data_parallel_group())
            if not totals[2].item():
                raise ValueError("GRPO evaluation produced no completions")
            metrics = {
                "eval/mean_reward": (totals[0] / totals[2].clamp(min=1)).item(),
                "eval/positive_reward_rate": (totals[1] / totals[2].clamp(min=1)).item(),
            }
            self._validate_distributed_loss(metrics["eval/mean_reward"], global_step)
            if is_first_rank():
                self.logger.info(f"GRPO evaluation at step {global_step}: {metrics}")
                for name, value in metrics.items():
                    log_metric(name, value, global_step)
            return metrics
        finally:
            self.model.train(was_training)

    def _finalize_process(self, destroy_process_group: bool = True):
        if self.reward_worker is not None:
            self.reward_worker.shutdown()
            self.reward_worker = None
        super()._finalize_process(destroy_process_group=destroy_process_group)
