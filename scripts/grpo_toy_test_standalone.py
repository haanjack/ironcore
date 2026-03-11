#!/usr/bin/env python
# Copyright (c) 2025-2026 Jaegeun Han
#
# SPDX-License-Identifier: Apache-2.0

"""
GRPO Toy Test: Train model to generate "ironcore" in outputs.

Objective:
    - Reward: +1 if "ironcore" in completion, 0 otherwise
    - Target: Success within 100 steps
    - Track: mean_reward (0→1), kl_divergence

Usage:
    python scripts/grpo_toy_test.py [--steps 100] [--device cuda:0]
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.optim import AdamW


@dataclass
class ToyTestConfig:
    """Configuration for toy test."""
    model_name: str = "Qwen/Qwen2.5-0.5B"
    device: str = "cuda:0"
    dtype: str = "bfloat16"

    batch_size: int = 4
    group_size: int = 4
    num_steps: int = 100
    learning_rate: float = 5e-5  # Higher LR for faster learning
    beta: float = 0.01  # Very low KL penalty for faster learning

    max_new_tokens: int = 32
    temperature: float = 1.0
    top_p: float = 0.9

    eval_interval: int = 10
    log_interval: int = 5

    target_word: str = "ironcore"
    output_dir: str = "toy_test_output"
    seed: int = 42


@dataclass
class MetricsTracker:
    """Track training metrics."""
    steps: list[int] = field(default_factory=list)
    mean_rewards: list[float] = field(default_factory=list)
    kl_divergences: list[float] = field(default_factory=list)
    success_rates: list[float] = field(default_factory=list)
    sample_outputs: list[dict] = field(default_factory=list)

    def log(self, step: int, mean_reward: float, kl: float, success_rate: float, samples: list[dict] | None = None):
        self.steps.append(step)
        self.mean_rewards.append(mean_reward)
        self.kl_divergences.append(kl)
        self.success_rates.append(success_rate)
        if samples:
            self.sample_outputs.append({"step": step, "samples": samples})

    def save(self, path: Path):
        data = {
            "steps": self.steps,
            "mean_rewards": self.mean_rewards,
            "kl_divergences": self.kl_divergences,
            "success_rates": self.success_rates,
            "sample_outputs": self.sample_outputs,
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)


class IroncoreRewardFunction:
    """Reward function: +1 if 'ironcore' in completion, 0 otherwise."""

    def __init__(self, target_word: str = "ironcore"):
        self.target_word = target_word.lower()

    def compute(self, prompt: str, completion: str, metadata: dict) -> float:
        """Compute reward based on presence of target word."""
        completion_lower = completion.lower()
        if self.target_word in completion_lower:
            return 1.0
        return 0.0


def load_model(config: ToyTestConfig):
    """Load model and tokenizer."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"Loading model: {config.model_name}")

    tokenizer = AutoTokenizer.from_pretrained(config.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
    dtype = dtype_map.get(config.dtype, torch.bfloat16)

    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        torch_dtype=dtype,
        device_map=config.device,
        trust_remote_code=True,
    )

    ref_model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        torch_dtype=dtype,
        device_map=config.device,
        trust_remote_code=True,
    )
    ref_model.eval()
    for param in ref_model.parameters():
        param.requires_grad = False

    return model, ref_model, tokenizer


def create_prompts(num_samples: int = 100) -> list[str]:
    """Create diverse prompts that encourage model to respond with 'ironcore'."""
    templates = [
        "Tell me about ironcore.",
        "What is ironcore?",
        "Describe the ironcore framework.",
        "What do you know about ironcore?",
        "Explain ironcore to me.",
        "Tell me about the ironcore project.",
        "What is the ironcore framework?",
        "Describe ironcore.",
        "What can you tell me about ironcore?",
        "Tell me what ironcore is.",
        "Explain the ironcore project.",
        "What is ironcore used for?",
        "Tell me about ironcore framework.",
        "Describe what ironcore does.",
        "What should I know about ironcore?",
    ]

    prompts = []
    for i in range(num_samples):
        prompts.append(templates[i % len(templates)])
    return prompts


def generate_completions(
    model: nn.Module,
    tokenizer,
    prompts: list[str],
    max_new_tokens: int = 32,
    temperature: float = 1.0,
    top_p: float = 0.9,
    do_sample: bool = True,
) -> list[str]:
    """Generate completions for a batch of prompts."""
    model.eval()
    completions = []

    with torch.no_grad():
        for prompt in prompts:
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=256)
            inputs = {k: v.to(model.device) for k, v in inputs.items()}

            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature if do_sample else 1.0,
                top_p=top_p if do_sample else 1.0,
                do_sample=do_sample,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

            # Decode only the new tokens
            new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
            completion = tokenizer.decode(new_tokens, skip_special_tokens=True)
            completions.append(completion)

    model.train()
    return completions


def compute_log_probs(model: nn.Module, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Compute log probabilities for sequences."""
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits

        # Shift for next token prediction
        shift_logits = logits[:, :-1, :]
        shift_labels = input_ids[:, 1:]

        # Compute log probs
        log_probs = torch.log_softmax(shift_logits.float(), dim=-1)
        token_log_probs = log_probs.gather(2, shift_labels.unsqueeze(-1)).squeeze(-1)

        # Mask padding
        shift_mask = attention_mask[:, 1:]
        token_log_probs = token_log_probs * shift_mask

        return token_log_probs.sum(dim=1)


def compute_kl_divergence(policy_log_probs: torch.Tensor, ref_log_probs: torch.Tensor) -> torch.Tensor:
    """Compute KL divergence."""
    # KL(ref || policy)
    kl = ref_log_probs.exp() * (ref_log_probs - policy_log_probs)
    return kl


def run_toy_test(config: ToyTestConfig) -> dict[str, Any]:
    """Run the GRPO toy test."""
    print("=" * 70)
    print("GRPO Toy Test: Train model to generate 'ironcore'")
    print("=" * 70)
    print(f"Config: {config}")
    print()

    # Setup
    torch.manual_seed(config.seed)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    print("[1/4] Loading model...")
    model, ref_model, tokenizer = load_model(config)

    # Setup reward function
    reward_fn = IroncoreRewardFunction(config.target_word)

    # Create prompts
    print("[2/4] Creating prompts...")
    all_prompts = create_prompts(num_samples=config.batch_size * config.num_steps * 2)

    # Setup optimizer
    optimizer = AdamW(model.parameters(), lr=config.learning_rate)

    # Metrics tracker
    tracker = MetricsTracker()

    # Training loop
    print(f"[3/4] Running {config.num_steps} training steps...")
    print("-" * 70)

    prompt_idx = 0
    success_at_step = None

    for step in range(1, config.num_steps + 1):
        # Sample prompts
        batch_prompts = all_prompts[prompt_idx:prompt_idx + config.batch_size]
        prompt_idx = (prompt_idx + config.batch_size) % len(all_prompts)

        # Generate completions for each prompt (group_size times)
        all_completions = []
        all_rewards = []
        all_group_ids = []

        for g in range(config.group_size):
            completions = generate_completions(
                model, tokenizer, batch_prompts,
                max_new_tokens=config.max_new_tokens,
                temperature=config.temperature,
                top_p=config.top_p,
                do_sample=True,
            )
            all_completions.extend(completions)

            # Compute rewards
            for prompt, completion in zip(batch_prompts, completions, strict=False):
                reward = reward_fn.compute(prompt, completion, {})
                all_rewards.append(reward)

        # Create group IDs
        for b in range(config.batch_size):
            all_group_ids.extend([b] * config.group_size)

        rewards = torch.tensor(all_rewards, dtype=torch.float32, device=config.device)
        group_ids = torch.tensor(all_group_ids, dtype=torch.long, device=config.device)

        # Compute advantages (group normalization)
        advantages = torch.zeros_like(rewards)
        for g in group_ids.unique():
            mask = group_ids == g
            group_rewards = rewards[mask]
            if len(group_rewards) > 1:
                mean = group_rewards.mean()
                std = group_rewards.std()
                if std > 1e-8:
                    advantages[mask] = (group_rewards - mean) / std

        # Compute log probs for GRPO loss (simplified)
        # Tokenize completions
        full_texts = [p + c for p, c in zip(
            batch_prompts * config.group_size, all_completions, strict=False
        )]
        encoded = tokenizer(
            full_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=256,
        )
        input_ids = encoded["input_ids"].to(config.device)
        attention_mask = encoded["attention_mask"].to(config.device)

        # Forward pass
        model.train()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits

        # Compute loss (simplified GRPO)
        shift_logits = logits[:, :-1, :]
        shift_labels = input_ids[:, 1:]
        shift_mask = attention_mask[:, 1:]

        log_probs = torch.log_softmax(shift_logits.float(), dim=-1)
        token_log_probs = log_probs.gather(2, shift_labels.unsqueeze(-1)).squeeze(-1)
        sequence_log_probs = (token_log_probs * shift_mask).sum(dim=1)

        # Policy loss: -mean(advantage * log_prob)
        policy_loss = -(advantages.detach() * sequence_log_probs).mean()

        # KL loss (simplified: just regularize toward ref)
        with torch.no_grad():
            ref_outputs = ref_model(input_ids=input_ids, attention_mask=attention_mask)
            ref_log_probs = torch.log_softmax(ref_outputs.logits[:, :-1, :].float(), dim=-1)
            ref_token_log_probs = ref_log_probs.gather(2, shift_labels.unsqueeze(-1)).squeeze(-1)
            ref_sequence_log_probs = (ref_token_log_probs * shift_mask).sum(dim=1)

        kl_per_seq = (ref_sequence_log_probs.exp() * (ref_sequence_log_probs - sequence_log_probs))
        kl_loss = config.beta * kl_per_seq.mean()

        # Total loss
        loss = policy_loss + kl_loss

        # Backward
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        # Metrics
        mean_reward = rewards.mean().item()
        kl_div = kl_per_seq.mean().item()
        success_rate = rewards.mean().item()  # Same as mean_reward for binary

        tracker.log(step, mean_reward, kl_div, success_rate)

        # Log
        if step % config.log_interval == 0:
            # Audit sample outputs
            samples = []
            for i in range(min(3, len(batch_prompts))):
                prompt = batch_prompts[i]
                completion = all_completions[i]
                reward = all_rewards[i]
                has_target = config.target_word.lower() in completion.lower()
                samples.append({
                    "prompt": prompt[:50] + "...",
                    "completion": completion[:100] + "...",
                    "reward": reward,
                    f"has_{config.target_word}": has_target,
                })

            tracker.sample_outputs.append({"step": step, "samples": samples})

            print(
                f"  Step {step:3d} | "
                f"reward: {mean_reward:.3f} | "
                f"kl: {kl_div:.4f} | "
                f"loss: {loss.item():.4f}"
            )

            for s in samples:
                status = "✓" if s[f"has_{config.target_word}"] else "✗"
                print(f"    {status} {s['completion'][:60]}...")

        # Check success
        if step % config.eval_interval == 0:
            print(f"\n  [Eval @ {step}] Success rate: {success_rate:.1%}")
            if success_rate >= 0.8 and success_at_step is None:
                success_at_step = step
                print(f"  ★ SUCCESS! Model learned to generate '{config.target_word}' at step {step}")

    # Final analysis
    print("\n[4/4] Analyzing results...")
    print("-" * 70)

    # Save metrics
    tracker.save(output_dir / "metrics.json")

    # Summary
    print("\n" + "=" * 70)
    print("TOY TEST SUMMARY")
    print("=" * 70)

    print(f"\nTarget word: '{config.target_word}'")
    print(f"Steps run: {config.num_steps}")

    print("\nMean Reward Trend:")
    for i in range(0, len(tracker.steps), max(1, len(tracker.steps) // 10)):
        step = tracker.steps[i]
        reward = tracker.mean_rewards[i]
        kl = tracker.kl_divergences[i]
        bar = "█" * int(reward * 20)
        print(f"  Step {step:3d}: {reward:.2f} {bar:20s} KL={kl:.4f}")

    print(f"\nKL Divergence Range: [{min(tracker.kl_divergences):.4f}, {max(tracker.kl_divergences):.4f}]")

    # Final success rate
    final_success_rate = tracker.success_rates[-1]
    print(f"\nFinal Success Rate: {final_success_rate:.1%}")

    # Check if test passed
    passed = success_at_step is not None or final_success_rate >= 0.5

    print("-" * 70)
    if passed:
        if success_at_step:
            print(f"✓ PASS: Model learned to generate '{config.target_word}' at step {success_at_step}")
        else:
            print(f"✓ PASS: Final success rate {final_success_rate:.1%} >= 50%")
    else:
        print(f"✗ FAIL: Model did not learn. Final success rate: {final_success_rate:.1%}")

    print(f"\nResults saved to: {output_dir}")

    return {
        "passed": passed,
        "success_at_step": success_at_step,
        "final_success_rate": final_success_rate,
        "metrics": {
            "steps": tracker.steps,
            "mean_rewards": tracker.mean_rewards,
            "kl_divergences": tracker.kl_divergences,
        },
    }


def main():
    parser = argparse.ArgumentParser(description="GRPO Toy Test")
    parser.add_argument("--steps", type=int, default=100, help="Number of training steps")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    parser.add_argument("--group-size", type=int, default=4, help="Group size")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-0.5B", help="Model name")
    parser.add_argument("--target", type=str, default="ironcore", help="Target word to learn")
    parser.add_argument("--output-dir", type=str, default="toy_test_output", help="Output directory")
    args = parser.parse_args()

    config = ToyTestConfig(
        model_name=args.model,
        device=args.device,
        batch_size=args.batch_size,
        group_size=args.group_size,
        num_steps=args.steps,
        target_word=args.target,
        output_dir=args.output_dir,
    )

    results = run_toy_test(config)
    sys.exit(0 if results["passed"] else 1)


if __name__ == "__main__":
    main()
