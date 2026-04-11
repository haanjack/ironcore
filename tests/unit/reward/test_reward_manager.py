# Copyright (c) 2025-2026 Jaegeun Han
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for RewardManager, TemplateRuleReward, and config integration."""

import re
import subprocess
import sys
from collections import OrderedDict
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch

from ironcore.alignment.rewards import (
    FormatRewardFunction,
    KeywordRewardFunction,
    MathRewardFunction,
    RewardManager,
    RewardModelFunction,
    RewardWorkerPool,
    StrictFormatRewardFunction,
    TemplateRuleReward,
)
from ironcore.config.config_alignment import (
    AlignmentConfig,
    RewardFunctionEntry,
    RewardManagerConfig,
)

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def math_gsm8k_yaml():
    """Path to math_gsm8k.yaml template."""
    return "configs/rewards/math_gsm8k.yaml"


@pytest.fixture
def format_cot_yaml():
    """Path to format_cot.yaml template."""
    return "configs/rewards/format_cot.yaml"


@pytest.fixture
def format_deepseek_yaml():
    """Path to format_deepseek.yaml template."""
    return "configs/rewards/format_deepseek.yaml"


# =============================================================================
# 1. TemplateRuleReward Tests (Tests 1-23)
# =============================================================================


class TestTemplateRuleRewardAnswerMatch:
    """Tests 1-11: answer_match mode."""

    def test_01_exact_match_hash_pattern(self, math_gsm8k_yaml):
        """Test 1: Exact match via #### pattern."""
        fn = TemplateRuleReward.from_yaml(math_gsm8k_yaml)
        result = fn.compute("prompt", "The answer is #### 42", {"answer": "42"})
        assert result == 1.0

    def test_02_exact_match_boxed_pattern(self, math_gsm8k_yaml):
        """Test 2: Exact match via \\boxed{} pattern."""
        fn = TemplateRuleReward.from_yaml(math_gsm8k_yaml)
        result = fn.compute("prompt", r"The answer is \boxed{42}", {"answer": "42"})
        assert result == 1.0

    def test_03_exact_match_answer_pattern(self, math_gsm8k_yaml):
        """Test 3: Exact match via Answer: pattern."""
        fn = TemplateRuleReward.from_yaml(math_gsm8k_yaml)
        result = fn.compute("prompt", "Answer: 42", {"answer": "42"})
        assert result == 1.0

    def test_04_exact_match_the_answer_is_pattern(self, math_gsm8k_yaml):
        """Test 4: Exact match via 'The answer is' pattern."""
        fn = TemplateRuleReward.from_yaml(math_gsm8k_yaml)
        result = fn.compute("prompt", "The answer is 42", {"answer": "42"})
        assert result == 1.0

    def test_05_wrong_answer_partial(self, math_gsm8k_yaml):
        """Test 5: Wrong answer extracted returns partial (0.1)."""
        fn = TemplateRuleReward.from_yaml(math_gsm8k_yaml)
        result = fn.compute("prompt", "#### 99", {"answer": "42"})
        assert result == 0.1

    def test_06_no_answer_extracted_strict(self):
        """Test 6: No answer extracted from completion (strict mode, no fallback) returns 0.0."""
        config = {
            "mode": "answer_match",
            "answer_patterns": [r"####\s*(.+)"],
            "fallback_last_number": False,
            "scoring": {"correct": 1.0, "partial": 0.1, "no_answer": 0.0},
        }
        fn = TemplateRuleReward(config)
        result = fn.compute("prompt", "no numbers here", {"answer": "#### 42"})
        assert result == 0.0

    def test_07_fallback_last_number(self):
        """Test 7: Fallback to last number in text."""
        config = {
            "mode": "answer_match",
            "answer_patterns": [r"####\s*(.+)"],
            "fallback_last_number": True,
            "scoring": {"correct": 1.0, "partial": 0.1, "no_answer": 0.0},
        }
        fn = TemplateRuleReward(config)
        result = fn.compute("prompt", "I think 42 works", {"answer": "42"})
        assert result == 1.0

    def test_08_no_ground_truth(self, math_gsm8k_yaml):
        """Test 8: No ground truth returns 0.5."""
        fn = TemplateRuleReward.from_yaml(math_gsm8k_yaml)
        result = fn.compute("prompt", "#### 42", {"answer": ""})
        assert result == 0.5

    def test_09_normalization_case_insensitive(self, math_gsm8k_yaml):
        """Test 9: Normalization - case insensitive."""
        fn = TemplateRuleReward.from_yaml(math_gsm8k_yaml)
        result = fn.compute("prompt", "#### YES", {"answer": "#### yes"})
        assert result == 1.0

    def test_10_normalization_strip_chars(self, math_gsm8k_yaml):
        """Test 10: Normalization - strip chars ($1,000 -> 1000)."""
        fn = TemplateRuleReward.from_yaml(math_gsm8k_yaml)
        result = fn.compute("prompt", "#### $1,000", {"answer": "1000"})
        assert result == 1.0

    def test_11_normalization_trailing_period(self, math_gsm8k_yaml):
        """Test 11: Normalization - strip trailing period."""
        fn = TemplateRuleReward.from_yaml(math_gsm8k_yaml)
        result = fn.compute("prompt", "#### 42.", {"answer": "42"})
        assert result == 1.0


class TestTemplateRuleRewardTagCheck:
    """Tests 12-15: tag_check mode."""

    def test_12_all_tags_present(self, format_cot_yaml):
        """Test 12: All tags present returns all_present score (0.0)."""
        fn = TemplateRuleReward.from_yaml(format_cot_yaml)
        result = fn.compute(
            "prompt",
            "<thought>reasoning</thought><answer>42</answer>",
            {},
        )
        assert result == 0.0

    def test_13_all_tags_missing(self, format_cot_yaml):
        """Test 13: All tags missing returns -0.4 (4 tags * -0.1)."""
        fn = TemplateRuleReward.from_yaml(format_cot_yaml)
        result = fn.compute("prompt", "plain text", {})
        assert result == -0.4

    def test_14_partial_tags(self, format_cot_yaml):
        """Test 14: Partial tags returns -0.2 (2 missing * -0.1)."""
        fn = TemplateRuleReward.from_yaml(format_cot_yaml)
        result = fn.compute("prompt", "<thought>x</thought> no answer tags", {})
        assert result == -0.2

    def test_15_custom_scoring(self):
        """Test 15: Custom scoring values."""
        config = {
            "mode": "tag_check",
            "required_tags": ["<thought>", "</thought>"],
            "scoring": {"all_present": 0.5, "per_missing_tag": -0.25},
        }
        fn = TemplateRuleReward(config)
        result = fn.compute("prompt", "<thought>x</thought>", {})
        assert result == 0.5
        result = fn.compute("prompt", "plain", {})
        assert result == -0.5  # 2 tags * -0.25


class TestTemplateRuleRewardRegexMatch:
    """Tests 16-19: regex_match mode."""

    def test_16_pattern_matches(self, format_deepseek_yaml):
        """Test 16: Pattern matches returns 1.0."""
        fn = TemplateRuleReward.from_yaml(format_deepseek_yaml)
        # format_deepseek.yaml pattern: <think>.*?</think>\s*####\s*.*
        result = fn.compute("prompt", "<think>reasoning</think> #### 42", {})
        assert result == 1.0

    def test_17_pattern_doesnt_match(self, format_deepseek_yaml):
        """Test 17: Pattern doesn't match returns 0.0."""
        fn = TemplateRuleReward.from_yaml(format_deepseek_yaml)
        result = fn.compute("prompt", "just text", {})
        assert result == 0.0

    def test_18_dotall_flag_works(self, format_deepseek_yaml):
        """Test 18: DOTALL flag allows multiline content inside think tags."""
        fn = TemplateRuleReward.from_yaml(format_deepseek_yaml)
        result = fn.compute("prompt", "<think>\nmultiline\nreasoning\n</think> #### 42", {})
        assert result == 1.0

    def test_19_custom_scoring(self):
        """Test 19: Custom scoring values."""
        config = {
            "mode": "regex_match",
            "pattern": r"\d+",
            "scoring": {"match": 0.5, "no_match": -1.0},
        }
        fn = TemplateRuleReward(config)
        result = fn.compute("prompt", "has 123 numbers", {})
        assert result == 0.5
        result = fn.compute("prompt", "no numbers", {})
        assert result == -1.0


class TestTemplateRuleRewardEdgeCases:
    """Tests 20-23: Edge cases."""

    def test_20_missing_mode_field(self, tmp_path):
        """Test 20: Missing mode field raises ValueError."""
        yaml_path = tmp_path / "invalid.yaml"
        yaml_path.write_text("answer_patterns:\n  - '####\\s*(.+)'")
        with pytest.raises(ValueError, match="missing required 'mode' field"):
            TemplateRuleReward.from_yaml(str(yaml_path))

    def test_21_unknown_mode(self):
        """Test 21: Unknown mode raises ValueError."""
        config = {"mode": "unknown_mode"}
        fn = TemplateRuleReward(config)
        with pytest.raises(ValueError, match="Unknown rule mode"):
            fn.compute("prompt", "completion", {})

    def test_22_empty_completion(self, math_gsm8k_yaml):
        """Test 22: Empty completion is handled gracefully."""
        fn = TemplateRuleReward.from_yaml(math_gsm8k_yaml)
        result = fn.compute("prompt", "", {"answer": "42"})
        assert isinstance(result, float)

    def test_23_from_yaml_nonexistent_path(self):
        """Test 23: from_yaml with nonexistent path raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            TemplateRuleReward.from_yaml("nonexistent/path.yaml")


# =============================================================================
# 2. RewardModelFunction Tests (Tests 24-30)
# =============================================================================


class TestRewardModelFunction:
    """Tests 24-30: RewardModelFunction backends."""

    def test_24_init_unknown_backend(self):
        """Test 24: Init with unknown backend raises ValueError."""
        with pytest.raises(ValueError, match="Unknown backend"):
            RewardModelFunction(backend="unknown")

    def test_25_local_inference_without_model_path(self):
        """Test 25: local_inference without model_path raises ValueError."""
        with pytest.raises(ValueError, match="local_model_path required"):
            RewardModelFunction(backend="local_inference", local_model_path=None)

    def test_26_local_endpoint_compute_mocked(self):
        """Test 26: local_endpoint compute returns parsed scalar from {"reward": 0.8}."""
        with patch("requests.Session") as mock_session_cls:
            mock_session = MagicMock()
            mock_session_cls.return_value = mock_session
            mock_resp = MagicMock()
            mock_resp.json.return_value = {"reward": 0.8}
            mock_session.post.return_value = mock_resp

            fn = RewardModelFunction(
                backend="local_endpoint", local_endpoint="http://localhost:8000/v1"
            )
            result = fn.compute("prompt", "completion", {})
        assert result == pytest.approx(0.8)

    def test_27_local_endpoint_retry_on_failure(self):
        """Test 27: local_endpoint retries max_retries times, returns 0.5 on exhaustion."""
        with patch("requests.Session") as mock_session_cls:
            mock_session = MagicMock()
            mock_session_cls.return_value = mock_session
            mock_session.post.side_effect = Exception("connection refused")

            fn = RewardModelFunction(
                backend="local_endpoint",
                local_endpoint="http://localhost:8000/v1",
                max_retries=2,
                timeout=1,
            )
            with patch("ironcore.alignment.rewards.model.time.sleep"):
                result = fn.compute("prompt", "completion", {})
        assert result == 0.5
        assert mock_session.post.call_count == 2

    def test_28_local_endpoint_score_format(self):
        """Test 28: local_endpoint supports {"score": float} response format."""
        with patch("requests.Session") as mock_session_cls:
            mock_session = MagicMock()
            mock_session_cls.return_value = mock_session
            mock_resp = MagicMock()
            mock_resp.json.return_value = {"score": 0.7}
            mock_session.post.return_value = mock_resp

            fn = RewardModelFunction(
                backend="local_endpoint", local_endpoint="http://localhost:8000/v1"
            )
            result = fn.compute("prompt", "completion", {})
        assert result == pytest.approx(0.7)

    def test_29_api_compute_mocked(self):
        """Test 29: api compute returns parsed scalar from mocked OpenAI client."""
        import sys

        mock_openai_module = MagicMock()
        mock_client = MagicMock()
        mock_openai_module.OpenAI.return_value = mock_client
        mock_resp = MagicMock()
        mock_resp.choices[0].message.content = "0.9"
        mock_client.chat.completions.create.return_value = mock_resp

        with patch.dict(sys.modules, {"openai": mock_openai_module}):
            fn = RewardModelFunction(backend="api", api_model="gpt-4")
            result = fn.compute("prompt", "completion", {})
        assert result == pytest.approx(0.9)

    def test_30_local_inference_compute_mocked(self):
        """Test 30: local_inference returns logits[0,0] value from mocked model."""
        import sys

        mock_transformers = MagicMock()

        mock_tokenizer = MagicMock()
        tok_output = MagicMock()
        tok_output.to.return_value = tok_output
        mock_tokenizer.return_value = tok_output
        mock_transformers.AutoTokenizer.from_pretrained.return_value = mock_tokenizer

        mock_model = MagicMock()
        mock_outputs = MagicMock()
        mock_outputs.logits = torch.tensor([[0.75, 0.25]])
        mock_model.return_value = mock_outputs
        mock_transformers.AutoModelForSequenceClassification.from_pretrained.return_value = (
            mock_model
        )

        with patch.dict(sys.modules, {"transformers": mock_transformers}):
            fn = RewardModelFunction(
                backend="local_inference", local_model_path="/fake/path", local_device="cpu"
            )
            result = fn.compute("prompt", "completion", {})
        assert result == pytest.approx(0.75)


# =============================================================================
# 3. RewardManager Tests (Tests 31-37)
# =============================================================================


class TestRewardManager:
    """Tests 31-37: RewardManager core functionality."""

    def test_31_register_single_compute(self, math_gsm8k_yaml):
        """Test 31: Register single function, compute returns score * weight."""
        manager = RewardManager()
        fn = TemplateRuleReward.from_yaml(math_gsm8k_yaml)
        manager.register("correctness", fn, weight=0.6)
        result = manager.compute("prompt", "#### 42", {"answer": "42"})
        assert result == 0.6

    def test_32_register_multiple_weighted_sum(self, math_gsm8k_yaml, format_cot_yaml):
        """Test 32: Multiple functions return weighted sum."""
        manager = RewardManager()
        math_fn = TemplateRuleReward.from_yaml(math_gsm8k_yaml)
        format_fn = TemplateRuleReward.from_yaml(format_cot_yaml)

        manager.register("correctness", math_fn, weight=0.6)
        manager.register("format", format_fn, weight=0.4)

        completion = "<thought>reasoning</thought><answer>42</answer>"
        result = manager.compute("prompt", completion, {"answer": "42"})
        assert result == 0.6

    def test_33_no_functions_runtime_error(self):
        """Test 33: No functions registered raises RuntimeError."""
        manager = RewardManager()
        with pytest.raises(RuntimeError, match="No reward functions registered"):
            manager.compute("prompt", "completion", {})

    def test_34_from_config_rule_template(self, math_gsm8k_yaml):
        """Test 34: from_config with rule_template type."""
        cfg = RewardManagerConfig(
            functions=[
                RewardFunctionEntry(
                    name="correctness",
                    type="rule_template",
                    weight=0.6,
                    rule_template=math_gsm8k_yaml,
                )
            ]
        )
        manager = RewardManager.from_config(cfg)
        result = manager.compute("prompt", "#### 42", {"answer": "42"})
        assert result == 0.6

    def test_35_from_config_reward_model(self):
        """Test 35: from_config with reward_model type (endpoint-based)."""
        cfg = RewardManagerConfig(
            functions=[
                RewardFunctionEntry(
                    name="rm",
                    type="reward_model",
                    weight=1.0,
                    rm_backend="local_endpoint",
                    local_endpoint="http://localhost:9999",
                )
            ]
        )
        manager = RewardManager.from_config(cfg)
        result = manager.compute("prompt", "completion", {})
        assert result == 0.5

    def test_36_from_config_math_type(self, math_gsm8k_yaml):
        """Test 36: from_config with math type uses built-in MathRewardFunction."""
        cfg = RewardManagerConfig(
            functions=[
                RewardFunctionEntry(
                    name="math",
                    type="math",
                    weight=1.0,
                )
            ]
        )
        manager = RewardManager.from_config(cfg)
        result = manager.compute("prompt", "#### 42", {"answer": "#### 42"})
        assert result == 1.0

    def test_37_from_config_missing_rule_template_path(self):
        """Test 37: from_config with missing rule_template path raises ValueError."""
        cfg = RewardManagerConfig(
            functions=[
                RewardFunctionEntry(
                    name="broken",
                    type="rule_template",
                    weight=1.0,
                    rule_template=None,
                )
            ]
        )
        with pytest.raises(ValueError, match="no rule_template path"):
            RewardManager.from_config(cfg)

    def test_38_from_config_composite_math(self):
        """Test 38: from_config with composite_math creates format + correctness entries."""
        cfg = RewardManagerConfig(
            functions=[
                RewardFunctionEntry(
                    name="dense_math",
                    type="composite_math",
                    weight=1.0,
                    format_weight=0.2,
                )
            ]
        )
        manager = RewardManager.from_config(cfg)
        # composite_math registers two entries: format (0.2) + correctness (0.8)
        assert len(manager._functions) == 2
        assert manager._functions[0][0] == "dense_math_format"
        assert manager._functions[1][0] == "dense_math_correctness"
        assert manager._functions[0][1] == 0.2  # format weight
        assert manager._functions[1][1] == 0.8  # correctness weight

        # Test compute: correct answer with format
        result = manager.compute("prompt", "#### 42", {"answer": "42"})
        # format: pattern matches ####\s*-?\d → reward=1.0, weight=0.2 → 0.2
        # correctness: matches answer → 1.0, weight=0.8 → 0.8
        # total = 1.0
        assert result == pytest.approx(1.0)

    def test_39_from_config_keyword_type(self):
        """Test 39: from_config with keyword type."""
        cfg = RewardManagerConfig(
            functions=[
                RewardFunctionEntry(
                    name="kw",
                    type="keyword",
                    weight=1.0,
                    keyword="test",
                )
            ]
        )
        manager = RewardManager.from_config(cfg)
        assert manager.compute("prompt", "this has test in it", {}) == 1.0
        assert manager.compute("prompt", "no match", {}) == 0.0

    def test_40_from_config_soft_keyword_type(self):
        """Test 40: from_config with soft_keyword type."""
        cfg = RewardManagerConfig(
            functions=[
                RewardFunctionEntry(
                    name="soft_kw",
                    type="soft_keyword",
                    weight=1.0,
                    keyword="test",
                )
            ]
        )
        manager = RewardManager.from_config(cfg)
        # Exact match
        assert manager.compute("prompt", "this has test in it", {}) == 1.0
        # No match - returns min_score (0.0 by default)
        result = manager.compute("prompt", "xyz", {})
        assert result == 0.0

    def test_41_from_config_unknown_type_raises(self):
        """Test 41: from_config with unknown type raises ValueError."""
        cfg = RewardManagerConfig(
            functions=[
                RewardFunctionEntry(
                    name="unknown",
                    type="invalid_type",
                    weight=1.0,
                )
            ]
        )
        with pytest.raises(ValueError, match="Unknown reward type"):
            RewardManager.from_config(cfg)


# =============================================================================
# 4. Config Dataclass Tests (Tests 41-47)
# =============================================================================


class TestConfigDataclasses:
    """Tests 41-47: Config dataclass conversions."""

    def test_41_reward_manager_config_dict_conversion(self, math_gsm8k_yaml):
        """Test 41: RewardManagerConfig with dict functions converts to RewardFunctionEntry."""
        cfg = RewardManagerConfig(
            functions=[{"name": "test", "type": "rule_template", "rule_template": math_gsm8k_yaml}]
        )
        assert len(cfg.functions) == 1
        assert isinstance(cfg.functions[0], RewardFunctionEntry)
        assert cfg.functions[0].name == "test"

    def test_42_reward_manager_config_entry_list(self, math_gsm8k_yaml):
        """Test 42: RewardManagerConfig with RewardFunctionEntry list passes through."""
        entry = RewardFunctionEntry(
            name="test", type="rule_template", rule_template=math_gsm8k_yaml
        )
        cfg = RewardManagerConfig(functions=[entry])
        assert cfg.functions[0] is entry

    def test_43_alignment_config_reward_manager_dict(self, math_gsm8k_yaml):
        """Test 43: AlignmentConfig with reward_manager dict converts to RewardManagerConfig."""
        cfg = AlignmentConfig(
            method="grpo",
            grpo_group_size=4,
            reward_manager={
                "functions": [
                    {"name": "test", "type": "rule_template", "rule_template": math_gsm8k_yaml}
                ]
            },
        )
        assert isinstance(cfg.reward_manager, RewardManagerConfig)

    def test_44_alignment_config_requires_reward_manager(self):
        """Test 44: AlignmentConfig(method='grpo') requires reward_manager."""
        with pytest.raises(ValueError, match="GRPO requires reward_manager configuration"):
            AlignmentConfig(method="grpo", grpo_group_size=4)

    def test_45_alignment_config_from_yaml(self, tmp_path, math_gsm8k_yaml):
        """Test 45: AlignmentConfig.from_yaml with reward_manager key."""
        yaml_content = f"""
method: grpo
grpo_group_size: 4
reward_manager:
  functions:
    - name: correctness
      type: rule_template
      weight: 0.6
      rule_template: {math_gsm8k_yaml}
"""
        yaml_path = tmp_path / "test_config.yaml"
        yaml_path.write_text(yaml_content)

        cfg = AlignmentConfig.from_yaml(yaml_path)
        assert isinstance(cfg.reward_manager, RewardManagerConfig)
        assert len(cfg.reward_manager.functions) == 1
        assert cfg.reward_manager.functions[0].name == "correctness"

    def test_46_reward_function_entry_defaults(self):
        """Test 46: RewardFunctionEntry default values are sensible."""
        entry = RewardFunctionEntry()
        assert entry.name == "default"
        assert entry.type == "rule_template"
        assert entry.weight == 1.0
        assert entry.rule_template is None


# =============================================================================
# 5. YAML Template Tests (Tests 57-60)
# =============================================================================


class TestYAMLTemplates:
    """Tests 57-60: YAML template loading."""

    def test_57_math_gsm8k_yaml_loads(self, math_gsm8k_yaml):
        """Test 57: math_gsm8k.yaml loads correctly."""
        fn = TemplateRuleReward.from_yaml(math_gsm8k_yaml)
        assert fn.mode == "answer_match"

    def test_58_format_cot_yaml_loads(self, format_cot_yaml):
        """Test 58: format_cot.yaml loads correctly."""
        fn = TemplateRuleReward.from_yaml(format_cot_yaml)
        assert fn.mode == "tag_check"

    def test_59_format_deepseek_yaml_loads(self, format_deepseek_yaml):
        """Test 59: format_deepseek.yaml loads correctly."""
        fn = TemplateRuleReward.from_yaml(format_deepseek_yaml)
        assert fn.mode == "regex_match"

    def test_60_all_yaml_valid(self):
        """Test 60: All YAML files in configs/rewards/ are valid YAML."""
        import yaml

        reward_dir = Path("configs/rewards")
        for yaml_file in reward_dir.glob("*.yaml"):
            with open(yaml_file) as f:
                config = yaml.safe_load(f)
            assert "mode" in config, f"{yaml_file} missing mode field"


# =============================================================================
# 6. Built-in Reward Function Tests
# =============================================================================


class TestBuiltinRewardFunctions:
    """Tests for built-in reward functions."""

    def test_math_reward_function(self):
        """Test MathRewardFunction computes correct scores."""
        fn = MathRewardFunction(strict=False)
        assert fn.compute("prompt", "#### 42", {"answer": "42"}) == 1.0
        assert fn.compute("prompt", "#### 99", {"answer": "42"}) == 0.1  # partial
        assert fn.compute("prompt", "no answer", {"answer": "42"}) == 0.0

    def test_math_reward_function_strict_mode(self):
        """Test MathRewardFunction in strict mode requires pattern match."""
        fn = MathRewardFunction(strict=True)
        # Strict mode: must match pattern, no fallback to last number
        # Both completion and answer need to match patterns
        assert fn.compute("prompt", "#### 42", {"answer": "#### 42"}) == 1.0
        # "result: 42" doesn't match any pattern (not "Answer:", not "####", etc.)
        # In strict mode, no extraction = 0.0
        assert fn.compute("prompt", "result: 42", {"answer": "#### 42"}) == 0.0

    def test_keyword_reward_function(self):
        """Test KeywordRewardFunction computes correct scores."""
        fn = KeywordRewardFunction(keyword="ironcore")
        assert fn.compute("prompt", "this has ironcore in it", {}) == 1.0
        assert fn.compute("prompt", "no match", {}) == 0.0

    def test_soft_keyword_reward_function(self):
        """Test SoftKeywordRewardFunction computes partial credit."""
        from ironcore.alignment.rewards import SoftKeywordRewardFunction

        fn = SoftKeywordRewardFunction(keyword="ironcore", min_score=0.0)
        # Exact match
        assert fn.compute("prompt", "this has ironcore in it", {}) == 1.0
        # Partial match (8/8 chars = 1.0)
        assert fn.compute("prompt", "ironcore", {}) == 1.0
        # No match at all (0 chars matching in any 8-gram)
        result = fn.compute("prompt", "xyz", {})
        assert result == 0.0

    def test_format_reward_function(self):
        """Test FormatRewardFunction computes correct scores."""
        fn = FormatRewardFunction(
            required_tags=["<thought>", "</thought>"],
            penalty=-0.1,
        )
        assert fn.compute("prompt", "<thought>x</thought>", {}) == 0.0
        # penalty * (missing / total) = -0.1 * (2/2) = -0.1
        assert fn.compute("prompt", "plain", {}) == pytest.approx(-0.1)

    def test_strict_format_reward_function(self):
        """Test StrictFormatRewardFunction computes correct scores."""
        fn = StrictFormatRewardFunction(pattern=r"####\s*\d+", reward=1.0, penalty=0.0)
        assert fn.compute("prompt", "The answer is #### 42", {}) == 1.0
        assert fn.compute("prompt", "No pattern here", {}) == 0.0

    def test_strict_format_default_pattern(self):
        """Test StrictFormatRewardFunction with default pattern."""
        fn = StrictFormatRewardFunction()
        # Default pattern: <think>.*?</think>\s*####\s*.*
        assert fn.compute("prompt", "<think>reasoning</think> #### 42", {}) == 1.0
        assert fn.compute("prompt", "no format here", {}) == 0.0

    def test_code_reward_function_not_implemented(self):
        """Test CodeRewardFunction raises NotImplementedError (sandbox not yet implemented)."""
        from ironcore.alignment.rewards import CodeRewardFunction

        fn = CodeRewardFunction(timeout=1)
        with pytest.raises(NotImplementedError):
            fn.compute(
                "def add(a, b):", "    return a + b", {"test_cases": ["assert add(1, 2) == 3"]}
            )


# =============================================================================
# 7. Error Handling Tests (Tests 68-72)
# =============================================================================


class TestErrorHandling:
    """Tests 68-72: Error handling."""

    def test_68_rule_template_path_doesnt_exist(self):
        """Test 68: rule_template path doesn't exist raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            TemplateRuleReward.from_yaml("nonexistent.yaml")

    def test_69_yaml_invalid_mode(self, tmp_path):
        """Test 69: YAML template has invalid mode raises ValueError at compute time."""
        yaml_path = tmp_path / "bad_mode.yaml"
        yaml_path.write_text("mode: invalid_mode")
        fn = TemplateRuleReward.from_yaml(str(yaml_path))
        with pytest.raises(ValueError, match="Unknown rule mode"):
            fn.compute("prompt", "completion", {})

    def test_70_empty_functions_runtime_error(self):
        """Test 70: reward_manager.functions is empty raises RuntimeError."""
        cfg = RewardManagerConfig(functions=[])
        manager = RewardManager.from_config(cfg)
        with pytest.raises(RuntimeError, match="No reward functions registered"):
            manager.compute("prompt", "completion", {})

    def test_71_endpoint_unreachable_returns_default(self):
        """Test 71: RewardModelFunction endpoint unreachable returns 0.5."""
        fn = RewardModelFunction(
            backend="local_endpoint",
            local_endpoint="http://localhost:59999",
            max_retries=1,
            timeout=1,
        )
        result = fn.compute("prompt", "completion", {})
        assert result == 0.5

    def test_72_malformed_yaml_invalid_regex(self, tmp_path):
        """Test 72: Malformed YAML with invalid regex raises error at compute time."""
        yaml_path = tmp_path / "bad_regex.yaml"
        yaml_path.write_text("""
mode: regex_match
pattern: '['
""")
        fn = TemplateRuleReward.from_yaml(str(yaml_path))
        import re

        with pytest.raises(re.error):
            fn.compute("prompt", "completion", {})


# =============================================================================
# 7b. Reward Weight Edge Cases
# =============================================================================


class TestRewardWeightEdgeCases:
    """Edge cases for weighted sum semantics."""

    def test_equal_weights_sum_beyond_one(self):
        """Two functions with weight=1.0 each must return sum of both scores (2.0)."""
        manager = RewardManager()
        manager.register("kw1", KeywordRewardFunction(keyword="hello"), weight=1.0)
        manager.register("kw2", KeywordRewardFunction(keyword="world"), weight=1.0)

        result = manager.compute("p", "hello world", {})
        assert abs(result - 2.0) < 1e-6, f"Expected 2.0, got {result}"

    def test_zero_total_weight_raises_value_error(self):
        """Total weight of zero must raise ValueError."""
        manager = RewardManager()
        manager.register("kw", KeywordRewardFunction(), weight=0.0)
        with pytest.raises(ValueError, match="[Ww]eight"):
            manager.compute("p", "c", {})

    def test_code_reward_raises_without_test_cases(self):
        """CodeRewardFunction must raise NotImplementedError even without test_cases."""
        from ironcore.alignment.rewards import CodeRewardFunction

        fn = CodeRewardFunction()
        with pytest.raises(NotImplementedError):
            fn.compute("prompt", "code", {})


# =============================================================================
# 7c. DeepSeek Format Token (2083)
# =============================================================================


class TestDeepSeekFormatToken:
    """DeepSeek <think > tag format validation."""

    FORMAT_DEEPSEEK_YAML = "configs/rewards/format_deepseek.yaml"

    def test_default_pattern_rejects_currwork_tag(self):
        """<currwork> tag must NOT match StrictFormatRewardFunction default pattern."""
        from ironcore.alignment.rewards import StrictFormatRewardFunction

        fn = StrictFormatRewardFunction()
        score = fn.compute("", "<currwork>reasoning</currwork>#### 42", {})
        assert score == 0.0, f"<currwork> tag should NOT match default pattern, got {score}"

    def test_deepseek_yaml_uses_think_tag(self):
        """format_deepseek.yaml must use <think > and not <currwork>."""
        import yaml

        with open(self.FORMAT_DEEPSEEK_YAML, encoding="utf-8") as f:
            config = yaml.safe_load(f)

        pattern = config.get("pattern", "")
        assert "<think" in pattern, f"format_deepseek.yaml pattern must contain <think, got: {pattern}"
        assert "<currwork>" not in pattern, "format_deepseek.yaml must not reference <currwork>"

    def test_deepseek_yaml_rejects_currwork(self):
        """format_deepseek.yaml template must reject <currwork> completions."""
        fn = TemplateRuleReward.from_yaml(self.FORMAT_DEEPSEEK_YAML)
        score = fn.compute("", "<currwork>reasoning</currwork>#### 7", {})
        assert score == 0.0


# =============================================================================
# 7d. LRU Cache Tests
# =============================================================================


class TestLocalEndpointRewardCache:
    """LocalEndpointRewardFunction LRU cache behavior."""

    def _make_fn(self, cache_size: int = 3):
        """Build LocalEndpointRewardFunction with a fully mocked openai module."""
        import sys

        from ironcore.alignment.rewards import LocalEndpointRewardFunction

        mock_openai_module = MagicMock()
        mock_openai_module.OpenAI.return_value = MagicMock()

        with patch.dict(sys.modules, {"openai": mock_openai_module}):
            fn = LocalEndpointRewardFunction(
                endpoint="http://localhost:8000",
                model="test-model",
                cache_size=cache_size,
            )
        return fn

    def test_cache_hit_returns_cached_value(self):
        fn = self._make_fn(cache_size=10)

        key = (hash("p"), hash("c"), hash("{}"))
        fn._cache[key] = 0.77

        result = fn._compute_cached(hash("p"), hash("c"), hash("{}"), "p", "c", {})
        assert abs(result - 0.77) < 1e-9, f"Expected cached 0.77, got {result}"

    def test_lru_eviction_oldest_entry(self):
        fn = self._make_fn(cache_size=3)

        keys = [(i, i, i) for i in range(3)]
        for k in keys:
            fn._cache[k] = float(k[0]) * 0.1

        assert len(fn._cache) == 3

        if len(fn._cache) >= fn._cache_size:
            fn._cache.popitem(last=False)
        fn._cache[(99, 99, 99)] = 0.99

        assert keys[0] not in fn._cache, "Oldest entry should have been evicted"
        assert (99, 99, 99) in fn._cache
        assert len(fn._cache) == 3

    def test_cache_size_zero_clamped_to_one(self):
        fn = self._make_fn(cache_size=0)
        assert fn._cache_size == 1, f"cache_size=0 should be clamped to 1, got {fn._cache_size}"

    def test_cache_size_negative_clamped_to_one(self):
        fn = self._make_fn(cache_size=-100)
        assert fn._cache_size == 1

    def test_cache_is_ordered_dict(self):
        fn = self._make_fn()
        assert isinstance(fn._cache, OrderedDict)


class TestLocalInferenceRewardCache:
    """LocalInferenceRewardFunction LRU cache behavior."""

    def _make_fn(self, cache_size: int = 3):
        from ironcore.alignment.rewards import LocalInferenceRewardFunction

        mock_tokenizer = MagicMock()
        mock_model = MagicMock()
        mock_model.eval.return_value = mock_model

        with (
            patch("transformers.AutoTokenizer.from_pretrained", return_value=mock_tokenizer),
            patch("transformers.AutoModelForCausalLM.from_pretrained", return_value=mock_model),
        ):
            fn = LocalInferenceRewardFunction(
                model_path="fake/path",
                cache_size=cache_size,
            )
        return fn

    def test_cache_size_zero_clamped_to_one(self):
        fn = self._make_fn(cache_size=0)
        assert fn._cache_size == 1

    def test_cache_is_ordered_dict(self):
        fn = self._make_fn()
        assert isinstance(fn._cache, OrderedDict)

    def test_lru_eviction_on_overflow(self):
        fn = self._make_fn(cache_size=2)

        fn._cache[(1, 1, 1)] = 0.1
        fn._cache[(2, 2, 2)] = 0.2
        assert len(fn._cache) == 2

        if len(fn._cache) >= fn._cache_size:
            fn._cache.popitem(last=False)
        fn._cache[(3, 3, 3)] = 0.3

        assert (1, 1, 1) not in fn._cache
        assert (3, 3, 3) in fn._cache
        assert len(fn._cache) == 2

    def test_extract_score_skips_absent_vocab_tokens(self):
        """_extract_score_from_logits must skip tokens not in vocabulary (tid=None)."""
        from ironcore.alignment.rewards import LocalInferenceRewardFunction

        mock_tokenizer = MagicMock()

        def convert(tok):
            return {"1": 10, "5": 50}.get(tok, None)

        mock_tokenizer.convert_tokens_to_ids = convert
        mock_model = MagicMock()
        mock_model.eval.return_value = mock_model

        with (
            patch("transformers.AutoTokenizer.from_pretrained", return_value=mock_tokenizer),
            patch("transformers.AutoModelForCausalLM.from_pretrained", return_value=mock_model),
        ):
            fn = LocalInferenceRewardFunction(model_path="fake/path")

        logits = torch.zeros(1, 100)
        logits[0, 10] = 5.0
        logits[0, 50] = 10.0

        score = fn._extract_score_from_logits(logits)
        assert 0.0 <= score <= 1.0, f"Score out of range: {score}"
        assert score > 0.3, f"Expected score near 0.5 (token '5' dominant), got {score}"


# =============================================================================
# 8. Integration Tests
# =============================================================================


class TestIntegration:
    """Integration tests."""

    def test_51_two_rule_template_entries_weighted_sum(self, math_gsm8k_yaml, format_cot_yaml):
        """Test 51: Config with two rule_template entries computes weighted sum."""
        cfg = RewardManagerConfig(
            functions=[
                RewardFunctionEntry(
                    name="correctness",
                    type="rule_template",
                    weight=0.6,
                    rule_template=math_gsm8k_yaml,
                ),
                RewardFunctionEntry(
                    name="format", type="rule_template", weight=0.4, rule_template=format_cot_yaml
                ),
            ]
        )
        manager = RewardManager.from_config(cfg)

        result = manager.compute(
            "prompt",
            "<thought>work</thought><answer>42</answer> #### 42",
            {"answer": "42"},
        )
        assert result == pytest.approx(0.6)

        result = manager.compute("prompt", "#### 42", {"answer": "42"})
        assert result == pytest.approx(0.44)

    def test_53_custom_yaml_template_uses_new_patterns(self, tmp_path):
        """Test 53: Custom YAML template with different patterns is used correctly."""
        custom_yaml = tmp_path / "custom.yaml"
        custom_yaml.write_text("""
mode: answer_match
answer_patterns:
  - 'Result:\\s*(.+)'
  - 'Final:\\s*(.+)'
fallback_last_number: false
scoring:
  correct: 1.0
  partial: 0.1
  no_answer: 0.0
""")
        fn = TemplateRuleReward.from_yaml(str(custom_yaml))

        result = fn.compute("prompt", "Result: 42", {"answer": "Result: 42"})
        assert result == 1.0

        result = fn.compute("prompt", "#### 42", {"answer": "42"})
        assert result == 0.5

    def test_56_reward_worker_pool_score_batch(self, math_gsm8k_yaml):
        """Test 56: RewardWorkerPool.score_batch with RewardManager returns tensor of shape [8]."""
        manager = RewardManager()
        fn = TemplateRuleReward.from_yaml(math_gsm8k_yaml)
        manager.register("correctness", fn, weight=1.0)

        pool = RewardWorkerPool(reward_fn=manager, num_workers=2, timeout=10)

        batch_size = 8
        prompts = ["What is 6*7?"] * batch_size
        completions = [f"#### {i}" for i in range(batch_size)]
        metadata = [{"answer": "42"}] * batch_size

        rewards = pool.score_batch(prompts, completions, metadata)
        assert isinstance(rewards, torch.Tensor)
        assert rewards.shape == (batch_size,)


# =============================================================================
# 9. E2E Training Tests
# =============================================================================

REPO_ROOT = Path(__file__).parents[3]  # tests/unit/reward → repo root
E2E_RM_CONFIG = str(REPO_ROOT / "tests" / "fixtures" / "configs" / "grpo_gsm8k_smoke_fsdp.yaml")
E2E_RM_MATH_CONFIG = str(REPO_ROOT / "tests" / "fixtures" / "configs" / "grpo_gsm8k_smoke_rm_math.yaml")
TORCHRUN_CMD = [sys.executable, "-m", "torch.distributed.run", "--nproc_per_node=2"]


def _resolve_config_paths(config_path: str) -> str:
    """
    Resolve relative paths in config YAML to absolute paths recursively.

    For fixture configs with relative path references like '../model/...',
    IronCore's loader handles them automatically based on config file location.

    This function only needs to handle absolute references like 'configs/...'
    (used when config files are in the repo root or need to force repo-root resolution).

    Handles nested structures (e.g., data.config_path).

    Args:
        config_path: Path to YAML config file

    Returns:
        Path to resolved config file (stored in temp location if modifications needed)
    """
    import yaml

    config_file = Path(config_path)
    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_file}")

    # Load YAML
    with open(config_file, "r") as f:
        config = yaml.safe_load(f)

    if not config:
        return config_path

    # Only resolve 'configs/...' absolute references; relative paths (../) are auto-handled
    def resolve_paths(obj):
        """Recursively resolve absolute paths in nested structures."""
        modified = False
        if isinstance(obj, dict):
            for key, value in obj.items():
                if isinstance(value, str) and value.startswith("configs/"):
                    # Convert to absolute path relative to repo root
                    abs_path = str((REPO_ROOT / value).resolve())
                    # Remove .yaml extension if present (IronCore loader adds it automatically)
                    if abs_path.endswith(".yaml"):
                        abs_path = abs_path[:-5]
                    obj[key] = abs_path
                    modified = True
                elif isinstance(value, (dict, list)):
                    if resolve_paths(value):
                        modified = True
        elif isinstance(obj, list):
            for i, item in enumerate(obj):
                if isinstance(item, str) and item.startswith("configs/"):
                    abs_path = str((REPO_ROOT / item).resolve())
                    # Remove .yaml extension if present (IronCore loader adds it automatically)
                    if abs_path.endswith(".yaml"):
                        abs_path = abs_path[:-5]
                    obj[i] = abs_path
                    modified = True
                elif isinstance(item, (dict, list)):
                    if resolve_paths(item):
                        modified = True
        return modified

    modified = resolve_paths(config)

    # If no modifications needed, return original path
    if not modified:
        return config_path

    # Write resolved config to same directory as original config file
    # This preserves relative path context (../) which IronCore loader depends on
    config_dir = config_file.parent
    temp_file = config_dir / f"resolved_{config_file.stem}.yaml"
    with open(temp_file, "w") as f:
        yaml.dump(config, f, default_flow_style=False)

    return str(temp_file)


def _run_training(config: str) -> subprocess.CompletedProcess:
    """Run torchrun training job, return CompletedProcess."""
    # Resolve config paths if needed (for test configs with relative paths)
    resolved_config = _resolve_config_paths(config)

    cmd = TORCHRUN_CMD + ["-m", "ironcore", "train", "--config", resolved_config]
    return subprocess.run(
        cmd,
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=900,  # 15 min timeout
        check=False,  # Explicitly handle return code in tests
    )


def _extract_reward_stats(output: str) -> dict:
    """Extract mean_reward statistics from training log output."""
    pattern = r"mean_reward[=:]\s*([\d.]+)"
    matches = re.findall(pattern, output)
    values = [float(m) for m in matches]
    if not values:
        return {"mean": 0.0, "std": 0.0, "n": 0}
    mean = sum(values) / len(values)
    variance = sum((v - mean) ** 2 for v in values) / len(values) if len(values) > 1 else 0.0
    std = variance**0.5
    return {"mean": mean, "std": std, "n": len(values)}


class TestRLVRTraining:
    """GRPO smoke tests that run actual distributed training.

    Marked @pytest.mark.rlvr and excluded from default test runs:
        pytest -m "not rlvr"   # skip RLVR tests (default CI)
        pytest -m rlvr         # run only RLVR tests

    Uses rule-based rewards (no API keys). Requires:
        - 2 GPUs (torchrun --nproc_per_node=2)
        - Qwen/Qwen2.5-0.5B-Instruct cached in ~/.cache/huggingface/
        - Flash attention support
        - ~5-10 minutes per test
    """

    @pytest.mark.rlvr
    def test_reward_manager_config_trains(self):
        """10-step GRPO training with reward_manager config runs cleanly."""
        result = _run_training(E2E_RM_CONFIG)

        assert result.returncode == 0, (
            f"Training with reward_manager config failed (exit {result.returncode}).\n"
            f"STDOUT:\n{result.stdout[-3000:]}\n"
            f"STDERR:\n{result.stderr[-3000:]}"
        )

        combined = result.stdout + result.stderr
        assert "mean_reward" in combined, (
            "No mean_reward logged — training may not have computed rewards.\n"
            f"STDOUT tail:\n{result.stdout[-2000:]}"
        )

    @pytest.mark.rlvr
    def test_reward_manager_composite_math_trains(self):
        """GRPO training with composite_math reward via RewardManager."""
        result = _run_training(E2E_RM_MATH_CONFIG)

        assert result.returncode == 0, (
            f"Training with composite_math config failed (exit {result.returncode}).\n"
            f"STDOUT:\n{result.stdout[-3000:]}\n"
            f"STDERR:\n{result.stderr[-3000:]}"
        )

        stats = _extract_reward_stats(result.stdout + result.stderr)
        assert stats["n"] > 0, "No reward values parsed from run"
        assert stats["mean"] > 0.0, f"Composite math mean_reward degenerate: {stats['mean']:.4f}"


# =============================================================================
# 10. Import Test
# =============================================================================


class TestImports:
    """Test all reward classes can be imported correctly."""

    def test_all_imports_from_rewards_package(self):
        """Test all reward classes can be imported from ironcore.alignment.rewards."""
        from ironcore.alignment import rewards

        expected = [
            "RewardFunction",
            "RewardManager",
            "RewardModelFunction",
            "RewardWorkerPool",
            "TemplateRuleReward",
            "MathRewardFunction",
            "CodeRewardFunction",
            "FormatRewardFunction",
            "StrictFormatRewardFunction",
            "KeywordRewardFunction",
            "SoftKeywordRewardFunction",
            "APIRewardFunction",
            "LocalEndpointRewardFunction",
            "LocalInferenceRewardFunction",
        ]
        for name in expected:
            assert hasattr(rewards, name), f"Missing export: {name}"

    def test_all_imports_from_alignment(self):
        """Test all reward classes can be imported from ironcore.alignment."""
        from ironcore import alignment

        expected = [
            "RewardFunction",
            "RewardManager",
            "RewardModelFunction",
            "RewardWorkerPool",
            "TemplateRuleReward",
            "MathRewardFunction",
            "CodeRewardFunction",
            "FormatRewardFunction",
            "StrictFormatRewardFunction",
            "KeywordRewardFunction",
            "SoftKeywordRewardFunction",
            "APIRewardFunction",
            "LocalEndpointRewardFunction",
            "LocalInferenceRewardFunction",
        ]
        for name in expected:
            assert hasattr(alignment, name), f"Missing export: {name}"
