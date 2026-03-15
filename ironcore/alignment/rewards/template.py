# Copyright (c) 2025-2026 Jaegeun Han
#
# SPDX-License-Identifier: Apache-2.0

"""YAML-driven rule-based reward functions.

Externalizes answer extraction patterns, format checks, and regex matching
into user-editable YAML templates, replacing hardcoded Python logic.
"""

from __future__ import annotations

import re

import yaml

from .base import RewardFunction


class TemplateRuleReward(RewardFunction):
    """YAML-driven rule reward. Supports answer_match, tag_check, and regex_match modes."""

    def __init__(self, config: dict):
        self.mode = config["mode"]
        self.config = config

    @classmethod
    def from_yaml(cls, path: str) -> TemplateRuleReward:
        """Load a rule template from a YAML file."""
        with open(path, encoding="utf-8") as f:
            config = yaml.safe_load(f)
        if "mode" not in config:
            raise ValueError(f"Rule template {path} missing required 'mode' field")
        return cls(config)

    def compute(self, prompt: str, completion: str, metadata: dict) -> float:
        if self.mode == "answer_match":
            return self._answer_match(completion, metadata)
        if self.mode == "tag_check":
            return self._tag_check(completion)
        if self.mode == "regex_match":
            return self._regex_match(completion)
        raise ValueError(f"Unknown rule mode: {self.mode}")

    def _answer_match(self, completion: str, metadata: dict) -> float:
        """Extract answer from completion, normalize, compare to ground truth."""
        answer = metadata.get("answer", "")
        if not answer:
            return 0.5

        scoring = self.config.get("scoring", {})
        correct_score = scoring.get("correct", 1.0)
        partial_score = scoring.get("partial", 0.1)
        no_answer_score = scoring.get("no_answer", 0.0)

        extracted = self._extract_answer(completion)
        gold = self._extract_answer(answer)

        if not gold:
            return 0.5

        if not extracted:
            return no_answer_score

        if self._normalize(extracted) == self._normalize(gold):
            return correct_score

        return partial_score

    def _extract_answer(self, text: str) -> str:
        """Extract answer using configured patterns."""
        patterns = self.config.get("answer_patterns", [])
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                return match.group(1).strip()

        if self.config.get("fallback_last_number", False):
            numbers = re.findall(r"-?\d+\.?\d*", text)
            return numbers[-1] if numbers else ""

        return ""

    def _normalize(self, answer: str) -> str:
        """Normalize answer using configured rules."""
        norm_cfg = self.config.get("normalization", {})
        result = answer.strip()

        if norm_cfg.get("lowercase", False):
            result = result.lower()

        strip_chars = norm_cfg.get("strip_chars", "")
        if strip_chars:
            result = re.sub(f"[{re.escape(strip_chars)}]", "", result)

        if norm_cfg.get("strip_trailing_period", False) and result.endswith("."):
            result = result[:-1]

        return result

    def _tag_check(self, completion: str) -> float:
        """Check for required tags, apply penalty per missing tag."""
        required_tags = self.config.get("required_tags", [])
        scoring = self.config.get("scoring", {})
        all_present_score = scoring.get("all_present", 0.0)
        per_missing_penalty = scoring.get("per_missing_tag", -0.1)

        missing = sum(1 for tag in required_tags if tag not in completion)
        if missing == 0:
            return all_present_score
        return per_missing_penalty * missing

    def _regex_match(self, completion: str) -> float:
        """Full regex match, binary reward."""
        pattern = self.config.get("pattern", "")
        flags = 0
        for flag_name in self.config.get("pattern_flags", []):
            flags |= getattr(re, flag_name, 0)

        scoring = self.config.get("scoring", {})
        match_score = scoring.get("match", 1.0)
        no_match_score = scoring.get("no_match", 0.0)

        if re.search(pattern, completion, flags):
            return match_score
        return no_match_score
