# Copyright (c) 2025-2026 Jaegeun Han
#
# SPDX-License-Identifier: Apache-2.0

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

        super().__init__(task_name, split_name, tokenizer, batch_size, num_samples, cache_dir)

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
                expanded_labels.append(1 if choice_idx == int(examples["label"][i]) else 0)

        return {
            "prompts": expanded_prompts,
            "choices": expanded_choices,
            "labels": expanded_labels,
        }

    def _do_predict(
        self,
        model,
        tokenized_inputs,
        attention_mask: torch.Tensor | None = None,
        prompt_lens: list[int] | None = None,
    ) -> torch.tensor:
        # tokenized_inputs: [b, s]. Score each candidate ending on the
        # ending(+eos) tokens only: the shared prompt (ctx_a+ctx_b) and any
        # right-padding must be excluded, or a mean over the whole sequence
        # dilutes the comparison by however much prompt/padding each candidate
        # happens to carry (candidates have different ending lengths).
        input_ids = tokenized_inputs[:, :-1]
        labels = tokenized_inputs[:, 1:]

        # Get logits from model (no labels = inference mode)
        logits, _ = model(input_ids, labels=None)  # [b, s, v]

        batch_size, seq_len = labels.shape
        position_idx = torch.arange(seq_len, device=labels.device)
        per_sample_losses = []

        for i in range(batch_size):
            sample_logits = logits[i]  # [s, v]
            sample_labels = labels[i]  # [s]

            # attention_mask is over the *unshifted* sequence; shift it the same
            # way as labels (labels[j] = token[j+1]).
            if attention_mask is not None:
                valid = attention_mask[i, 1:].bool()
            else:
                valid = torch.ones(seq_len, dtype=torch.bool, device=labels.device)

            if prompt_lens is not None:
                # label[j] predicts token[j+1]; keep only j+1 >= prompt_len,
                # i.e. j >= prompt_len - 1.
                ending_start = max(prompt_lens[i] - 1, 0)
                valid = valid & (position_idx >= ending_start)

            per_token_loss = torch.nn.functional.cross_entropy(
                sample_logits, sample_labels, reduction="none"
            )
            valid_f = valid.float()
            if valid_f.sum() > 0:
                sample_loss = (per_token_loss * valid_f).sum() / valid_f.sum()
            else:
                # Should not normally happen (truncation could in principle
                # remove the whole ending) — fall back rather than div-by-zero.
                sample_loss = per_token_loss.mean()

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

        p_bar = None
        if is_first_rank():
            p_bar = tqdm(
                total=self.num_samples,
                bar_format="{l_bar}{bar} | {n:.0f}/{total:.0f} [{rate_fmt}]",
            )
        try:
            for batch in self.data_loader:
                torch.cuda.synchronize()

                prompts, choices, labels = self._get_batch(batch)

                # pad and tokenize inputs
                input_texts = []
                for prompt, choice in zip(prompts, choices, strict=True):
                    input_texts.append(prompt + " " + choice + self.tokenizer.eos_token)

                tokenized = self.tokenizer(
                    input_texts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    add_special_tokens=True,
                ).to(model.device)

                attention_mask = tokenized.get("attention_mask") if hasattr(
                    tokenized, "get"
                ) else None

                if "input_ids" in tokenized:
                    tokenized_inputs = tokenized["input_ids"]
                else:
                    tokenized_inputs = tokenized

                # Tokenize the shared prompt alone (no padding) to find where
                # each example's ending starts in the padded batch above.
                prompt_lens = [
                    len(ids)
                    for ids in self.tokenizer(
                        list(prompts), add_special_tokens=True, truncation=True
                    )["input_ids"]
                ]

                with torch.no_grad():
                    total_losses = self._do_predict(
                        model,
                        tokenized_inputs=tokenized_inputs,
                        attention_mask=attention_mask,
                        prompt_lens=prompt_lens,
                    )

                batch_prompts.extend(prompts)
                all_losses.extend(total_losses.tolist())
                all_labels.extend(labels)

                if p_bar is not None:
                    p_bar.update(self.num_samples / len(self.data_loader))
        finally:
            if p_bar is not None:
                p_bar.close()

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

            if 1 not in current_labels:
                raise ValueError(
                    f"HellaSwag example at prompt {total_samples} has no correct "
                    f"(label=1) ending among {len(current_labels)} candidates: "
                    f"{current_labels}. This indicates malformed data or that "
                    f"truncation split candidates from the same example across batches."
                )

            predicted_index = current_losses.index(min(current_losses))
            correct_index = current_labels.index(1)

            total_score += predicted_index == correct_index
            total_samples += 1

        # calculate overall score
        accuracy = total_score / total_samples * 100
        output = {"metric": "accuracy", "score": accuracy}
        return output
