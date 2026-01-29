# Copyright (c) 2025-2026 Jaegeun Han
#
# SPDX-License-Identifier: MIT
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the above copyright notice,
# this list of conditions, and the following disclaimer are retained.
#
# Full license text is available at LICENSE file.

import argparse
from abc import ABC
from pathlib import Path
from typing import Union

import torch
from tqdm import tqdm

from ironcore.utils import is_first_rank

from .base_task import Task


class HellaSwag(Task):
    def __init__(
        self,
        tokenizer,
        batch_size: int = 1,
        num_samples: int = None,
        cache_dir: Union[str, Path] = None,
    ):
        """
        initilize hellaswag benchmark
        """
        task_name = "hellaswag"
        split_name = "validation"

        super().__init__(
            task_name, split_name, tokenizer, batch_size, num_samples, cache_dir
        )

    @staticmethod
    def _preprocess(examples):
        expanded_prompts = []
        expanded_choices = []
        expanded_labels = []

        for i in range(len(examples["ctx_a"])):
            prompt = f"{examples['ctx_a'][i]} {examples['ctx_b'][i]}"
            for choice_idx, choice in enumerate(examples["endings"][i]):
                expanded_prompts.append(prompt)
                expanded_choices.append(choice)
                expanded_labels.append(
                    1 if choice_idx == int(examples["label"][i]) else 0
                )

        return {
            "prompts": expanded_prompts,
            "choices": expanded_choices,
            "labels": expanded_labels,
        }

    def _do_predict(self, model, tokenized_inputs) -> torch.tensor:

        # tokenized_inputs: [b, s]
        input_ids = tokenized_inputs[:, :-1]
        labels = tokenized_inputs[:, 1:]

        # Get logits from model (no labels = inference mode)
        logits = model(input_ids, labels=None)  # [b, s, v]

        # Compute per-sample losses manually
        # For each sample, compute average loss over its sequence
        batch_size = logits.size(0)
        per_sample_losses = []

        for i in range(batch_size):
            # Get loss for this sample's sequence
            sample_logits = logits[i]  # [s, v]
            sample_labels = labels[i]  # [s]

            # Compute cross-entropy loss for this sample
            sample_loss = torch.nn.functional.cross_entropy(
                sample_logits, sample_labels, reduction='mean'
            )
            per_sample_losses.append(sample_loss.item())

        return torch.tensor(per_sample_losses)

    def _get_batch(self, batch):
        """get evaluation context and label"""
        prompts, choices, labels = batch["prompts"], batch["choices"], batch["labels"]
        return prompts, choices, labels

    def process(self, model):
        """do eval"""
        model.eval()

        all_losses = []
        all_labels = []
        batch_prompts = []

        if is_first_rank():
            p_bar = tqdm(
                total=self.num_samples,
                bar_format="{l_bar}{bar} | {n:.0f}/{total:.0f} [{rate_fmt}]",
            )
        for batch in self.data_loader:
            torch.cuda.synchronize()

            prompts, choices, labels = self._get_batch(batch)

            # pad and tokenize inputs
            input_texts = []
            for prompt, choice in zip(prompts, choices):
                input_texts.append(prompt + " " + choice +
                                   self.tokenizer.eos_token)

            tokenized_inputs = self.tokenizer(
                input_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                add_special_tokens=True,
            ).to(model.device)

            if "input_ids" in tokenized_inputs:
                tokenized_inputs = tokenized_inputs["input_ids"]

            with torch.no_grad():
                total_losses = self._do_predict(
                    model, tokenized_inputs=tokenized_inputs
                )

            batch_prompts.extend(prompts)
            all_losses.extend(total_losses.tolist())
            all_labels.extend(labels)

            if is_first_rank():
                p_bar.update(self.num_samples / len(self.data_loader))

        return self._get_score(batch_prompts, all_losses, all_labels)

    def _get_score(self, batch_prompts, all_losses, all_labels) -> float:
        """get exact matching score"""
        total_score = 0
        total_samples = 0

        if len(batch_prompts) == 0:
            return 0.0

        i = 0
        while i < len(all_losses):
            current_prompt = batch_prompts[i]
            current_losses = []
            current_labels = []

            while i < len(batch_prompts) and batch_prompts[i] == current_prompt:
                current_losses.append(all_losses[i])
                current_labels.append(all_labels[i])
                i += 1

            predicted_index = current_losses.index(min(current_losses))
            correct_index = current_labels.index(1)

            total_score += predicted_index == correct_index
            total_samples += 1

        # calculate overall score
        accuracy = total_score / total_samples * 100
        output = {"metric": "accuracy", "score": accuracy}
        return output
