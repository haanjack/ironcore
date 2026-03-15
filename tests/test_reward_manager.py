# Copyright (c) 2025-2026 Jaegeun Han
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for RewardManager, TemplateRuleReward, and config integration."""

import warnings
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch

from ironcore.alignment.reward_manager import RewardManager
from ironcore.alignment.reward_model import RewardModelFunction
from ironcore.alignment.reward_rules import TemplateRuleReward
from ironcore.alignment.rewards import (
    CompositeRewardFunction,
    MathRewardFunction,
    RewardWorkerPool,
    get_reward_function,
)
from ironcore.config.config_alignment import (
    AlignmentConfig,
    RewardConfig,
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
        # When completion has no extractable answer, returns no_answer_score.
        # Ground truth extraction is used only for comparison, not as fallback.
        config = {
            "mode": "answer_match",
            "answer_patterns": [r"####\s*(.+)"],
            "fallback_last_number": False,
            "scoring": {"correct": 1.0, "partial": 0.1, "no_answer": 0.0},
        }
        fn = TemplateRuleReward(config)
        result = fn.compute("prompt", "no numbers here", {"answer": "#### 42"})
        assert result == 0.0  # No answer extracted from completion = no_answer

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
        # Both completion and answer must use extractable patterns
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
        """Test 18: DOTALL flag allows multiline content inside <think> tags."""
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
        # Should not crash
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

            fn = RewardModelFunction(backend="local_endpoint", local_endpoint="http://localhost:8000/v1")
            result = fn.compute("prompt", "completion", {})
        assert result == pytest.approx(0.8)

    def test_27_local_endpoint_retry_on_failure(self):
        """Test 27: local_endpoint retries max_retries times, returns 0.5 on exhaustion."""
        with patch("requests.Session") as mock_session_cls:
            mock_session = MagicMock()
            mock_session_cls.return_value = mock_session
            mock_session.post.side_effect = Exception("connection refused")

            fn = RewardModelFunction(backend="local_endpoint", local_endpoint="http://localhost:8000/v1", max_retries=2, timeout=1)
            # Patch time.sleep to avoid waiting
            with patch("ironcore.alignment.reward_model.time.sleep"):
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

            fn = RewardModelFunction(backend="local_endpoint", local_endpoint="http://localhost:8000/v1")
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
        # Tokenizer call returns a dict-like object with .to() that returns itself
        tok_output = MagicMock()
        tok_output.to.return_value = tok_output
        mock_tokenizer.return_value = tok_output
        mock_transformers.AutoTokenizer.from_pretrained.return_value = mock_tokenizer

        mock_model = MagicMock()
        mock_outputs = MagicMock()
        mock_outputs.logits = torch.tensor([[0.75, 0.25]])
        mock_model.return_value = mock_outputs
        mock_transformers.AutoModelForSequenceClassification.from_pretrained.return_value = mock_model

        with patch.dict(sys.modules, {"transformers": mock_transformers}):
            fn = RewardModelFunction(backend="local_inference", local_model_path="/fake/path", local_device="cpu")
            result = fn.compute("prompt", "completion", {})
        assert result == pytest.approx(0.75)


# =============================================================================
# 3. RewardManager Tests (Tests 31-40)
# =============================================================================

class TestRewardManager:
    """Tests 31-40: RewardManager core functionality."""

    def test_31_register_single_compute(self, math_gsm8k_yaml):
        """Test 31: Register single function, compute returns score * weight."""
        manager = RewardManager()
        fn = TemplateRuleReward.from_yaml(math_gsm8k_yaml)
        manager.register("correctness", fn, weight=0.6)
        result = manager.compute("prompt", "#### 42", {"answer": "42"})
        assert result == 0.6  # 1.0 * 0.6

    def test_32_register_multiple_weighted_sum(self, math_gsm8k_yaml, format_cot_yaml):
        """Test 32: Multiple functions return weighted sum."""
        manager = RewardManager()
        math_fn = TemplateRuleReward.from_yaml(math_gsm8k_yaml)
        format_fn = TemplateRuleReward.from_yaml(format_cot_yaml)  # Uses tag_check

        manager.register("correctness", math_fn, weight=0.6)
        manager.register("format", format_fn, weight=0.4)

        # Correct answer with all format tags present
        completion = "<thought>reasoning</thought><answer>42</answer>"
        result = manager.compute("prompt", completion, {"answer": "42"})
        # correctness: 1.0 * 0.6 = 0.6
        # format: 0.0 * 0.4 = 0.0 (all_present)
        # total: 0.6
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
                    local_endpoint="http://localhost:9999",  # Non-existent
                )
            ]
        )
        manager = RewardManager.from_config(cfg)
        # Will fail to connect but shouldn't crash
        result = manager.compute("prompt", "completion", {})
        # Returns default 0.5 after retries
        assert result == 0.5

    def test_36_from_config_legacy_type(self):
        """Test 36: from_config with legacy type delegates to get_reward_function()."""
        cfg = RewardManagerConfig(
            functions=[
                RewardFunctionEntry(
                    name="math",
                    type="math",
                    weight=1.0,
                )
            ]
        )
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            manager = RewardManager.from_config(cfg)
            # get_reward_function emits deprecation warning
            assert any("deprecated" in str(warning.message).lower() for warning in w)

        # Note: MathRewardFunction needs answer in extractable format too
        # Both completion and answer need to match patterns (e.g., #### prefix)
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
                    rule_template=None,  # Missing!
                )
            ]
        )
        with pytest.raises(ValueError, match="no rule_template path"):
            RewardManager.from_config(cfg)

    def test_38_from_legacy_config_math(self):
        """Test 38: from_legacy_config with type=math."""
        cfg = RewardConfig(type="math")
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            manager = RewardManager.from_legacy_config(cfg)
            assert any("deprecated" in str(warning.message).lower() for warning in w)

        # Note: MathRewardFunction needs answer in extractable format too
        result = manager.compute("prompt", "#### 42", {"answer": "#### 42"})
        assert result == 1.0

    def test_39_from_legacy_config_composite_math(self):
        """Test 39: from_legacy_config with type=composite_math."""
        cfg = RewardConfig(type="composite_math")
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            manager = RewardManager.from_legacy_config(cfg)
            # CompositeRewardFunction also emits warning
            assert len(w) >= 1

        result = manager.compute("prompt", "#### 42", {"answer": "42"})
        # Composite: format (0.2) + correctness (0.8) = varies based on format
        assert 0.0 <= result <= 1.0

    def test_40_from_legacy_config_keyword(self):
        """Test 40: from_legacy_config with type=keyword."""
        cfg = RewardConfig(type="keyword", keyword="test")
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            manager = RewardManager.from_legacy_config(cfg)
            assert any("deprecated" in str(warning.message).lower() for warning in w)

        result = manager.compute("prompt", "this has test in it", {})
        assert result == 1.0
        result = manager.compute("prompt", "no match", {})
        assert result == 0.0


# =============================================================================
# 4. Config Dataclass Tests (Tests 41-47)
# =============================================================================

class TestConfigDataclasses:
    """Tests 41-47: Config dataclass conversions."""

    def test_41_reward_manager_config_dict_conversion(self, math_gsm8k_yaml):
        """Test 41: RewardManagerConfig with dict functions converts to RewardFunctionEntry."""
        cfg = RewardManagerConfig(
            functions=[
                {"name": "test", "type": "rule_template", "rule_template": math_gsm8k_yaml}
            ]
        )
        assert len(cfg.functions) == 1
        assert isinstance(cfg.functions[0], RewardFunctionEntry)
        assert cfg.functions[0].name == "test"

    def test_42_reward_manager_config_entry_list(self, math_gsm8k_yaml):
        """Test 42: RewardManagerConfig with RewardFunctionEntry list passes through."""
        entry = RewardFunctionEntry(name="test", type="rule_template", rule_template=math_gsm8k_yaml)
        cfg = RewardManagerConfig(functions=[entry])
        assert cfg.functions[0] is entry

    def test_43_alignment_config_reward_manager_dict(self, math_gsm8k_yaml):
        """Test 43: AlignmentConfig with reward_manager dict converts to RewardManagerConfig."""
        cfg = AlignmentConfig(
            method="grpo",
            grpo_group_size=4,
            reward_manager={
                "functions": [{"name": "test", "type": "rule_template", "rule_template": math_gsm8k_yaml}]
            }
        )
        assert isinstance(cfg.reward_manager, RewardManagerConfig)

    def test_44_alignment_config_without_reward_manager(self):
        """Test 44: AlignmentConfig(method='grpo') without reward_manager validates reward.type."""
        # Default reward.type is "math" which is valid
        cfg = AlignmentConfig(method="grpo", grpo_group_size=4)
        assert cfg.reward.type == "math"

        # Invalid type should raise
        with pytest.raises(ValueError, match="reward.type"):
            AlignmentConfig(method="grpo", grpo_group_size=4, reward=RewardConfig(type="invalid"))

    def test_45_alignment_config_with_reward_manager_skips_validation(self, math_gsm8k_yaml):
        """Test 45: AlignmentConfig with reward_manager skips reward.type validation."""
        # This would normally fail because "invalid" isn't a valid type
        # But it's skipped because reward_manager is set
        cfg = AlignmentConfig(
            method="grpo",
            grpo_group_size=4,
            reward=RewardConfig(type="invalid"),
            reward_manager={
                "functions": [{"name": "test", "type": "rule_template", "rule_template": math_gsm8k_yaml}]
            }
        )
        assert cfg.reward_manager is not None

    def test_46_alignment_config_from_yaml(self, tmp_path, math_gsm8k_yaml):
        """Test 46: AlignmentConfig.from_yaml with reward_manager key."""
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

    def test_47_reward_function_entry_defaults(self):
        """Test 47: RewardFunctionEntry default values are sensible."""
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
# 6. Backward Compatibility Tests (Tests 48-50)
# =============================================================================

class TestBackwardCompatibility:
    """Tests 48-50: Legacy function compatibility."""

    def test_48_legacy_math_vs_template_identical(self, math_gsm8k_yaml):
        """Test 48: Legacy MathRewardFunction produces same scores as TemplateRuleReward."""
        legacy_fn = MathRewardFunction(strict=False)
        template_fn = TemplateRuleReward.from_yaml(math_gsm8k_yaml)

        test_cases = [
            ("prompt", "#### 42", {"answer": "42"}),
            ("prompt", "#### 99", {"answer": "42"}),
            ("prompt", "Answer: 100", {"answer": "100"}),
            ("prompt", r"\boxed{7}", {"answer": "7"}),
            ("prompt", "The answer is 15", {"answer": "15"}),
        ]

        for prompt, completion, metadata in test_cases:
            legacy_score = legacy_fn.compute(prompt, completion, metadata)
            template_score = template_fn.compute(prompt, completion, metadata)
            assert legacy_score == template_score, f"Mismatch for {completion}"

    def test_49_format_reward_different_formulas(self):
        """Test 49: FormatRewardFunction vs TemplateRuleReward have different penalty formulas.

        Legacy uses penalty * (missing / total_tags) while TemplateRuleReward uses per_missing_tag * count.
        This is a known design difference - users migrating should adjust per_missing_tag accordingly.
        """
        # Legacy
        from ironcore.alignment.rewards import FormatRewardFunction
        legacy_fn = FormatRewardFunction(
            required_tags=["<thought>", "</thought>", "<answer>", "</answer>"],
            penalty=-0.1,
        )

        # New via config
        config = {
            "mode": "tag_check",
            "required_tags": ["<thought>", "</thought>", "<answer>", "</answer>"],
            "scoring": {"all_present": 0.0, "per_missing_tag": -0.1},
        }
        template_fn = TemplateRuleReward(config)

        # Test all present - both return 0.0
        completion = "<thought>x</thought><answer>y</answer>"
        assert legacy_fn.compute("", completion, {}) == 0.0
        assert template_fn.compute("", completion, {}) == 0.0

        # Test partial - different formulas (documented)
        completion = "<thought>x</thought> no answer"
        legacy_result = legacy_fn.compute("", completion, {})
        template_result = template_fn.compute("", completion, {})
        # Legacy: penalty * (missing / total) = -0.1 * (2/4) = -0.05
        # Template: per_missing * count = -0.1 * 2 = -0.2
        assert legacy_result == pytest.approx(-0.05, abs=0.001)
        assert template_result == -0.2

    def test_50_strict_format_compatibility(self):
        """Test 50: StrictFormatRewardFunction behavior via regex_match."""
        from ironcore.alignment.rewards import StrictFormatRewardFunction

        legacy_fn = StrictFormatRewardFunction(pattern=r"####\s*\d+", reward=1.0, penalty=0.0)

        config = {
            "mode": "regex_match",
            "pattern": r"####\s*\d+",
            "scoring": {"match": 1.0, "no_match": 0.0},
        }
        template_fn = TemplateRuleReward(config)

        test_cases = [
            "The answer is #### 42",
            "No pattern here",
            "#### 100",
        ]

        for completion in test_cases:
            legacy_score = legacy_fn.compute("", completion, {})
            template_score = template_fn.compute("", completion, {})
            assert legacy_score == template_score, f"Mismatch for {completion}"


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
            local_endpoint="http://localhost:59999",  # Non-existent port
            max_retries=1,
            timeout=1,
        )
        result = fn.compute("prompt", "completion", {})
        assert result == 0.5

    def test_72_malformed_yaml_invalid_regex(self, tmp_path):
        """Test 72: Malformed YAML with invalid regex raises error at load time."""
        yaml_path = tmp_path / "bad_regex.yaml"
        yaml_path.write_text("""
mode: regex_match
pattern: '['
""")
        # Invalid regex should raise re.error when pattern is compiled
        # But TemplateRuleReward doesn't pre-compile, so error happens at compute
        fn = TemplateRuleReward.from_yaml(str(yaml_path))
        import re
        with pytest.raises(re.error):
            fn.compute("prompt", "completion", {})


# =============================================================================
# 8. Deprecation Warning Tests (Tests 61-64)
# =============================================================================

class TestDeprecationWarnings:
    """Tests 61-64: Deprecation warning emissions."""

    def test_61_get_reward_function_deprecation(self):
        """Test 61: get_reward_function() emits DeprecationWarning."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            get_reward_function("math")
            assert any(issubclass(warning.category, DeprecationWarning) for warning in w)

    def test_62_composite_reward_deprecation(self):
        """Test 62: CompositeRewardFunction emits DeprecationWarning."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            CompositeRewardFunction([(1.0, MathRewardFunction())])
            assert any(issubclass(warning.category, DeprecationWarning) for warning in w)

    def test_63_from_legacy_config_warning(self):
        """Test 63: from_legacy_config emits warning via get_reward_function."""
        cfg = RewardConfig(type="math")
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            RewardManager.from_legacy_config(cfg)
            assert any(issubclass(warning.category, DeprecationWarning) for warning in w)

    def test_64_from_config_no_deprecation(self, math_gsm8k_yaml):
        """Test 64: from_config with rule_template emits no deprecation warning."""
        cfg = RewardManagerConfig(
            functions=[
                RewardFunctionEntry(
                    name="test",
                    type="rule_template",
                    rule_template=math_gsm8k_yaml,
                )
            ]
        )
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            RewardManager.from_config(cfg)
            # No deprecation warnings
            assert not any(issubclass(warning.category, DeprecationWarning) for warning in w)


# =============================================================================
# 9. Integration Tests (Tests 51-56)
# =============================================================================


class TestNewConfigPath:
    """Tests 51-53: New config path integration."""

    def test_51_two_rule_template_entries_weighted_sum(self, math_gsm8k_yaml, format_cot_yaml):
        """Test 51: Config with two rule_template entries computes weighted sum."""
        cfg = RewardManagerConfig(
            functions=[
                RewardFunctionEntry(name="correctness", type="rule_template", weight=0.6, rule_template=math_gsm8k_yaml),
                RewardFunctionEntry(name="format", type="rule_template", weight=0.4, rule_template=format_cot_yaml),
            ]
        )
        manager = RewardManager.from_config(cfg)

        # Correct answer, format tags present
        result = manager.compute(
            "prompt",
            "<thought>work</thought><answer>42</answer> #### 42",
            {"answer": "42"},
        )
        # correctness=1.0*0.6=0.6, format=0.0*0.4=0.0 → 0.6
        assert result == pytest.approx(0.6)

        # Correct answer, format tags missing
        result = manager.compute("prompt", "#### 42", {"answer": "42"})
        # correctness=1.0*0.6=0.6, format=-0.4*0.4=-0.16 → 0.44
        assert result == pytest.approx(0.44)

    def test_52_mixed_rule_template_and_legacy(self, math_gsm8k_yaml):
        """Test 52: Config with mixed types (rule_template + legacy math) registers both."""
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            cfg = RewardManagerConfig(
                functions=[
                    RewardFunctionEntry(name="template", type="rule_template", weight=0.5, rule_template=math_gsm8k_yaml),
                    RewardFunctionEntry(name="legacy_math", type="math", weight=0.5),
                ]
            )
            manager = RewardManager.from_config(cfg)

        assert len(manager._functions) == 2
        result = manager.compute("prompt", "#### 42", {"answer": "#### 42"})
        assert isinstance(result, float)

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

        # Custom patterns match — answer also needs to match a pattern (gold extraction)
        result = fn.compute("prompt", "Result: 42", {"answer": "Result: 42"})
        assert result == 1.0

        # Standard #### pattern not in template; "42" also doesn't match any pattern
        # → gold extraction fails → returns 0.5 (ambiguous, design choice)
        result = fn.compute("prompt", "#### 42", {"answer": "42"})
        assert result == 0.5


class TestGRPOTrainerIntegration:
    """Tests 54-56: GRPOTrainer reward worker initialization."""

    def test_54_post_checkpoint_load_with_reward_manager(self, math_gsm8k_yaml):
        """Test 54: _post_checkpoint_load with reward_manager config initializes via from_config."""
        from ironcore.trainers.grpo_trainer import GRPOTrainer

        mock_config = MagicMock()
        mock_config.alignment.reward_manager = RewardManagerConfig(
            functions=[RewardFunctionEntry(name="correctness", type="rule_template", weight=1.0, rule_template=math_gsm8k_yaml)],
            num_workers=2,
            timeout=10,
        )
        mock_config.alignment.reward = RewardConfig(type="math")

        trainer = GRPOTrainer.__new__(GRPOTrainer)
        trainer.config = mock_config
        trainer.logger = MagicMock()
        trainer._reward_config = mock_config.alignment.reward

        with (
            patch.object(GRPOTrainer, "_create_reference_model", return_value=MagicMock()),
            patch.object(GRPOTrainer, "_setup_data_iterators"),
        ):
            trainer._post_checkpoint_load(last_step=0)

        assert trainer.reward_worker is not None
        assert isinstance(trainer.reward_worker.reward_fn, RewardManager)
        assert len(trainer.reward_worker.reward_fn._functions) == 1

    def test_55_post_checkpoint_load_with_legacy_config(self):
        """Test 55: _post_checkpoint_load with legacy config initializes via from_legacy_config."""
        from ironcore.trainers.grpo_trainer import GRPOTrainer

        mock_config = MagicMock()
        mock_config.alignment.reward_manager = None
        mock_config.alignment.reward = RewardConfig(type="math")

        trainer = GRPOTrainer.__new__(GRPOTrainer)
        trainer.config = mock_config
        trainer.logger = MagicMock()
        trainer._reward_config = mock_config.alignment.reward

        with (
            patch.object(GRPOTrainer, "_create_reference_model", return_value=MagicMock()),
            patch.object(GRPOTrainer, "_setup_data_iterators"),
            warnings.catch_warnings(record=True),
        ):
            warnings.simplefilter("always")
            trainer._post_checkpoint_load(last_step=0)

        assert trainer.reward_worker is not None
        assert isinstance(trainer.reward_worker.reward_fn, RewardManager)

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
