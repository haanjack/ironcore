#!/usr/bin/env python
# Copyright (c) 2025-2026 Jaegeun Han
#
# SPDX-License-Identifier: Apache-2.0

"""GRPO Keyword Insertion Test

Validates the GRPO pipeline by training GPT-2-small to include "ironcore" in responses.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ironcore.alignment.buffer import RolloutBuffer
from ironcore.alignment.loss.grpo import compute_advantages, grpo_loss
from ironcore.alignment.rewards import KeywordRewardFunction, RewardWorkerPool


# Configuration
PROMPTS = [
    "What is machine learning?",
    "Explain how neural networks work.",
    "What is the capital of France?",
    "Tell me about reinforcement learning.",
    "How does gradient descent work?",
    "What is Python used for?",
    "Describe the solar system.",
    "What is a transformer model?",
    "Why is the sky blue?",
    "What are the benefits of exercise?",
    "How does electricity work?",
    "What is object-oriented programming?",
    "Explain photosynthesis briefly.",
    "What is the theory of relativity?",
    "How do computers store data?",
    "What is climate change?",
]

KEYWORD = "ironcore"
GROUP_SIZE = 8
MAX_NEW_TOKENS = 128
TEMPERATURE = 0.8
TOP_P = 0.9
TOP_K = 50

BETA = 0.04
LR = 1e-6
TRAIN_STEPS = 300
BATCH_SIZE = 8
LOG_INTERVAL = 10
SAMPLE_LOG_INTERVAL = 50


class PromptDataset(Dataset):
    """Simple dataset that cycles through prompts."""

    def __init__(self, prompts: list[str], tokenizer, max_length: int = 256):
        self.prompts = prompts
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.prompts) * 1000  # Effectively infinite

    def __getitem__(self, idx):
        prompt = self.prompts[idx % len(self.prompts)]
        encoded = self.tokenizer(
            prompt,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "prompt": prompt,
            "metadata": {"keyword": KEYWORD, "type": "keyword"},
        }


def collate_fn(batch):
    return {
        "input_ids": torch.stack([b["input_ids"] for b in batch]),
        "attention_mask": torch.stack([b["attention_mask"] for b in batch]),
        "prompts": [b["prompt"] for b in batch],
        "metadata": [b["metadata"] for b in batch],
    }


@torch.no_grad()
def generate_completions(
    model: nn.Module,
    tokenizer,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    group_size: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Generate multiple completions per prompt using transformers generate."""
    B = input_ids.size(0)
    G = group_size

    # Expand batch: each prompt appears G times
    expanded_input_ids = input_ids.unsqueeze(1).expand(-1, G, -1).reshape(B * G, -1)
    expanded_attention_mask = attention_mask.unsqueeze(1).expand(-1, G, -1).reshape(B * G, -1)

    expanded_input_ids = expanded_input_ids.to(device)
    expanded_attention_mask = expanded_attention_mask.to(device)

    # Generate using transformers
    outputs = model.generate(
        input_ids=expanded_input_ids,
        attention_mask=expanded_attention_mask,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k if top_k > 0 else None,
        do_sample=True,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        return_dict_in_generate=True,
    )

    completion_ids = outputs.sequences
    generated_ids = completion_ids[:, input_ids.size(1):]

    # Compute log probs for each completion
    all_log_probs = []
    for i in range(0, len(completion_ids), B * G):
        batch_ids = completion_ids[i : i + B * G]
        labels = batch_ids.clone()
        labels[:, :-1] = batch_ids[:, 1:]
        labels[:, -1] = -100

        outputs = model(batch_ids, labels=labels)
        # Sum log probs for non-ignored tokens
        logits = outputs.logits[:, :-1, :]
        labels_shifted = labels[:, 1:]
        log_probs = torch.log_softmax(logits, dim=-1)
        token_log_probs = log_probs.gather(2, labels_shifted.unsqueeze(-1)).squeeze(-1)
        # Mask padding
        mask = (labels_shifted != -100).float()
        seq_log_probs = (token_log_probs * mask).sum(dim=-1)
        all_log_probs.append(seq_log_probs)

    log_probs_tensor = torch.cat(all_log_probs) if all_log_probs else torch.zeros(B * G, device=device)

    # Group IDs: [0,0,...,0, 1,1,...,1, ...]
    group_ids = torch.arange(B, device=device).unsqueeze(1).expand(-1, G).reshape(-1)

    return completion_ids, generated_ids, log_probs_tensor, group_ids


def compute_rewards(
    prompts: list[str],
    completions: list[str],
    metadata: list[dict],
    reward_fn: KeywordRewardFunction,
    worker_pool: RewardWorkerPool,
) -> torch.Tensor:
    """Compute rewards using worker pool."""
    # Repeat prompts G times to match completions
    G = len(completions) // len(prompts)
    repeated_prompts = [p for p in prompts for _ in range(G)]
    rewards = worker_pool.score_batch(
        prompts=repeated_prompts,
        completions=completions,
        metadata_list=metadata,
    )
    return rewards


def main():
    print("=" * 60)
    print("GRPO Keyword Insertion Test")
    print("=" * 60)
    print(f"Keyword: {KEYWORD}")
    print(f"Group size: {GROUP_SIZE}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Train steps: {TRAIN_STEPS}")
    print(f"KL coefficient: {BETA}")
    print(f"Learning rate: {LR}")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")

    # Load tokenizer and model
    print("\nLoading tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained("gpt2")
    model.to(device)
    model.train()

    # Create reference model (frozen)
    print("Creating reference model...")
    ref_model = AutoModelForCausalLM.from_pretrained("gpt2")
    ref_model.to(device)
    ref_model.eval()
    for param in ref_model.parameters():
        param.requires_grad = False

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

    # Dataset
    dataset = PromptDataset(PROMPTS, tokenizer)
    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        collate_fn=collate_fn,
        shuffle=True,
    )
    data_iter = iter(dataloader)

    # Reward function
    reward_fn = KeywordRewardFunction(keyword=KEYWORD)
    reward_pool = RewardWorkerPool(reward_fn, num_workers=4, timeout=30)

    # Tracking
    baseline_rate = 0.0
    hit_rates = []
    kl_values = []
    losses = []

    print("\n" + "=" * 60)
    print("Baseline Generation")
    print("=" * 60)

    # Generate baseline
    model.eval()
    batch = next(data_iter)
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    prompts = batch["prompts"]

    with torch.no_grad():
        completion_ids, generated_ids, log_probs, group_ids = generate_completions(
            model, tokenizer, input_ids, attention_mask,
            GROUP_SIZE, MAX_NEW_TOKENS, TEMPERATURE, TOP_P, TOP_K, device
        )

    completions = tokenizer.batch_decode(completion_ids, skip_special_tokens=True)
    hits = sum(1 for c in completions if KEYWORD in c.lower())
    baseline_rate = hits / len(completions)
    print(f"Baseline hit rate: {baseline_rate:.1%} ({hits}/{len(completions)})")
    hit_rates.append(baseline_rate)

    print("\n" + "=" * 60)
    print("Training")
    print("=" * 60)

    model.train()

    for step in range(1, TRAIN_STEPS + 1):
        # Get batch
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)

        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        prompts = batch["prompts"]
        metadata = batch["metadata"]

        # Generate completions
        model.eval()
        with torch.no_grad():
            completion_ids, generated_ids, old_log_probs, group_ids = generate_completions(
                model, tokenizer, input_ids, attention_mask,
                GROUP_SIZE, MAX_NEW_TOKENS, TEMPERATURE, TOP_P, TOP_K, device
            )
        model.train()

        # Decode for rewards
        completions = tokenizer.batch_decode(completion_ids, skip_special_tokens=True)
        expanded_prompts = [p for p in prompts for _ in range(GROUP_SIZE)]

        # Compute rewards
        rewards = compute_rewards(prompts, completions, metadata, reward_fn, reward_pool)
        rewards = rewards.to(device)

        # Compute advantages
        advantages = compute_advantages(rewards, group_ids, eps=1e-8, distributed=False)

        # Forward pass through both models
        labels = completion_ids.clone()
        labels[:, :-1] = completion_ids[:, 1:]
        labels[:, -1] = -100
        labels[:, :input_ids.size(1) - 1] = -100  # Mask prompt tokens

        # Policy log probs
        policy_outputs = model(completion_ids, labels=labels)
        policy_logits = policy_outputs.logits[:, :-1, :]
        labels_shifted = labels[:, 1:]
        policy_log_probs_full = torch.log_softmax(policy_logits, dim=-1)
        policy_token_logps = policy_log_probs_full.gather(2, labels_shifted.unsqueeze(-1)).squeeze(-1)
        mask = (labels_shifted != -100).float()
        policy_log_probs = (policy_token_logps * mask).sum(dim=-1)

        # Reference log probs (no grad)
        with torch.no_grad():
            ref_outputs = ref_model(completion_ids, labels=labels)
            ref_logits = ref_outputs.logits[:, :-1, :]
            ref_log_probs_full = torch.log_softmax(ref_logits, dim=-1)
            ref_token_logps = ref_log_probs_full.gather(2, labels_shifted.unsqueeze(-1)).squeeze(-1)
            ref_log_probs = (ref_token_logps * mask).sum(dim=-1)

        # Approximate KL per sequence
        kl_per_seq = (policy_token_logps - ref_token_logps) * mask
        kl_per_seq = kl_per_seq.sum(dim=-1)

        # GRPO loss
        loss, metrics = grpo_loss(
            policy_log_probs=policy_log_probs,
            ref_log_probs=ref_log_probs,
            advantages=advantages,
            kl_per_seq=kl_per_seq,
            beta=BETA,
            old_log_probs=old_log_probs,
            clip_eps=0.0,
        )

        # Backward
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        # Track metrics
        hit_rate = rewards.mean().item()
        hit_rates.append(hit_rate)
        kl_values.append(metrics["kl_per_seq"])
        losses.append(metrics["grpo_loss"])

        if step % LOG_INTERVAL == 0:
            print(f"step {step:4d}: loss={loss.item():.4f}, hit_rate={hit_rate:.1%}, kl={metrics['kl_per_seq']:.4f}")

        if step % SAMPLE_LOG_INTERVAL == 0:
            print(f"\n  Sample outputs at step {step}:")
            for i in range(min(3, len(completions))):
                hit = "✓" if KEYWORD in completions[i].lower() else "✗"
                print(f"    [{hit}] {completions[i][:100]}...")
            print()

    # Final evaluation
    print("\n" + "=" * 60)
    print("Final Evaluation")
    print("=" * 60)

    model.eval()
    total_hits = 0
    total_samples = 0

    with torch.no_grad():
        for _ in range(2):  # Generate a few more batches
            batch = next(data_iter)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            completion_ids, _, _, _ = generate_completions(
                model, tokenizer, input_ids, attention_mask,
                GROUP_SIZE, MAX_NEW_TOKENS, TEMPERATURE, TOP_P, TOP_K, device
            )

            completions = tokenizer.batch_decode(completion_ids, skip_special_tokens=True)
            hits = sum(1 for c in completions if KEYWORD in c.lower())
            total_hits += hits
            total_samples += len(completions)

            for i, c in enumerate(completions[:3]):
                hit = "✓" if KEYWORD in c.lower() else "✗"
                print(f"  [{hit}] {c[:100]}...")

    final_rate = total_hits / total_samples
    print(f"\nFinal hit rate: {final_rate:.1%} ({total_hits}/{total_samples})")

    # Summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"Baseline hit rate: {baseline_rate:.1%}")
    print(f"Final hit rate:    {final_rate:.1%}")
    print(f"Improvement:       {final_rate - baseline_rate:.1f}pp")
    print(f"Final KL:          {kl_values[-1]:.4f}")
    print(f"Final loss:        {losses[-1]:.4f}")

    success = final_rate >= 0.9
    print(f"\n{'✓ PASS' if success else '✗ FAIL'}: Keyword hit rate {'>' if success else '<'} 90%")

    reward_pool.shutdown()
    return success


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
