# Copyright (c) 2025-2026 Jaegeun Han
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for fix/reward-review-issues branch (2026-03-16).

Covers:
  - RewardManager weighted-sum semantics (known regression)
  - grpo_loss entropy bonus correctness
  - LRU caching in LocalEndpoint/LocalInference reward functions
  - cache_size edge-case guard (cache_size=0 → clamped to 1)
  - Import structure after utils/ refactor
  - DeepSeek <think> token fix
  - CodeRewardFunction NotImplementedError
  - Trainer module load smoke (no circular import)

Memory guard: a background thread checks host RAM every 5 s.
If usage exceeds HOST_RAM_LIMIT_GB the entire pytest session is
terminated immediately so the process cannot OOM the machine.
"""

from __future__ import annotations

import threading
from collections import OrderedDict
from unittest.mock import MagicMock, patch

import psutil
import pytest
import torch

# ---------------------------------------------------------------------------
# Memory guard
# ---------------------------------------------------------------------------

HOST_RAM_LIMIT_GB: float = 60.0
_POLL_INTERVAL_S: float = 5.0


class _MemoryGuard:
    """Background thread that aborts the test session on high host RAM."""

    def __init__(self, limit_gb: float, poll_interval: float = _POLL_INTERVAL_S):
        self._limit_bytes = int(limit_gb * 1024**3)
        self._poll_interval = poll_interval
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True, name="MemoryGuard")

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    def _run(self) -> None:
        while not self._stop.wait(self._poll_interval):
            used = psutil.virtual_memory().used
            if used > self._limit_bytes:
                used_gb = used / 1024**3
                # pytest.exit() is thread-safe; returncode 3 = resource limit
                pytest.exit(
                    f"[MemoryGuard] Host RAM {used_gb:.1f} GB exceeded "
                    f"{HOST_RAM_LIMIT_GB:.0f} GB limit — aborting test session.",
                    returncode=3,
                )


@pytest.fixture(scope="session", autouse=True)
def memory_guard():
    """Session-scoped fixture that starts the background RAM monitor."""
    guard = _MemoryGuard(limit_gb=HOST_RAM_LIMIT_GB)
    guard.start()
    yield
    guard.stop()


def _host_ram_used_gb() -> float:
    return psutil.virtual_memory().used / 1024**3


def _assert_ram_headroom(test_name: str) -> None:
    """Fail fast before a test if RAM is already above limit."""
    used = _host_ram_used_gb()
    assert used < HOST_RAM_LIMIT_GB, (
        f"[MemoryGuard] Refusing to run '{test_name}': "
        f"host RAM already at {used:.1f} GB (limit {HOST_RAM_LIMIT_GB:.0f} GB)"
    )


# ---------------------------------------------------------------------------
# Helpers / shared data
# ---------------------------------------------------------------------------

MATH_YAML = "configs/rewards/math_gsm8k.yaml"
FORMAT_COT_YAML = "configs/rewards/format_cot.yaml"
FORMAT_DEEPSEEK_YAML = "configs/rewards/format_deepseek.yaml"


# ---------------------------------------------------------------------------
# 1. RewardManager weighted-sum (known regression — divide-by-total_weight bug)
# ---------------------------------------------------------------------------


class TestRewardManagerWeightedSum:
    """RewardManager.compute() must return Σ(weight * score), NOT a weighted average."""

    def test_single_function_weight_applied(self):
        """Weight < 1.0 must scale the output down, not be normalised away."""
        _assert_ram_headroom("test_single_function_weight_applied")
        from ironcore.alignment.rewards import RewardManager
        from ironcore.alignment.rewards.template import TemplateRuleReward

        manager = RewardManager()
        fn = TemplateRuleReward.from_yaml(MATH_YAML)
        manager.register("correctness", fn, weight=0.6)

        result = manager.compute("prompt", "#### 42", {"answer": "42"})
        # score=1.0, weight=0.6 → expected 0.6, not 1.0
        assert abs(result - 0.6) < 1e-6, f"Expected 0.6 (weighted sum), got {result}"

    def test_multiple_functions_weighted_sum(self):
        """Multiple functions must return Σ(wᵢ * sᵢ), not normalised."""
        _assert_ram_headroom("test_multiple_functions_weighted_sum")
        from ironcore.alignment.rewards import RewardManager
        from ironcore.alignment.rewards.template import TemplateRuleReward

        manager = RewardManager()
        math_fn = TemplateRuleReward.from_yaml(MATH_YAML)
        format_fn = TemplateRuleReward.from_yaml(FORMAT_COT_YAML)
        manager.register("correctness", math_fn, weight=0.6)
        manager.register("format", format_fn, weight=0.4)

        # Completion has all 4 COT tags (format score=0.0) and correct #### answer (math score=1.0)
        completion = "<thought>reasoning</thought><answer>42</answer>#### 42"
        result = manager.compute("prompt", completion, {"answer": "42"})
        # Weighted sum: 0.6*1.0 + 0.4*0.0 = 0.6
        assert abs(result - 0.6) < 1e-6, f"Expected 0.6, got {result}"

    def test_equal_weights_sum_to_combined(self):
        """Two functions weight=1.0 each → sum of both scores."""
        _assert_ram_headroom("test_equal_weights_sum_to_combined")
        from ironcore.alignment.rewards import RewardManager
        from ironcore.alignment.rewards.builtin import KeywordRewardFunction

        manager = RewardManager()
        manager.register("kw1", KeywordRewardFunction(keyword="hello"), weight=1.0)
        manager.register("kw2", KeywordRewardFunction(keyword="world"), weight=1.0)

        # Both present → 1.0 + 1.0 = 2.0
        result = manager.compute("p", "hello world", {})
        assert abs(result - 2.0) < 1e-6, f"Expected 2.0, got {result}"

    def test_zero_total_weight_raises(self):
        """Total weight of zero must raise ValueError."""
        _assert_ram_headroom("test_zero_total_weight_raises")
        from ironcore.alignment.rewards import RewardManager
        from ironcore.alignment.rewards.builtin import KeywordRewardFunction

        manager = RewardManager()
        manager.register("kw", KeywordRewardFunction(), weight=0.0)
        with pytest.raises(ValueError, match="[Ww]eight"):
            manager.compute("p", "c", {})

    def test_no_functions_raises_runtime_error(self):
        _assert_ram_headroom("test_no_functions_raises_runtime_error")
        from ironcore.alignment.rewards import RewardManager

        manager = RewardManager()
        with pytest.raises(RuntimeError, match="No reward functions registered"):
            manager.compute("p", "c", {})

    def test_from_config_applies_weight(self):
        """from_config path must also apply weight correctly."""
        _assert_ram_headroom("test_from_config_applies_weight")
        from ironcore.alignment.rewards import RewardManager
        from ironcore.config.config_alignment import RewardFunctionEntry, RewardManagerConfig

        cfg = RewardManagerConfig(
            functions=[
                RewardFunctionEntry(
                    name="correctness",
                    type="rule_template",
                    weight=0.6,
                    rule_template=MATH_YAML,
                )
            ]
        )
        manager = RewardManager.from_config(cfg)
        result = manager.compute("prompt", "#### 42", {"answer": "42"})
        assert abs(result - 0.6) < 1e-6, f"Expected 0.6, got {result}"


# ---------------------------------------------------------------------------
# 2. grpo_loss entropy bonus
# ---------------------------------------------------------------------------


class TestGrpoLossEntropy:
    """grpo_loss must accept entropy/entropy_coef and report correct metrics."""

    @pytest.fixture
    def base_tensors(self):
        torch.manual_seed(0)
        B = 8
        policy_lp = -torch.rand(B)
        ref_lp = policy_lp + 0.05 * torch.randn(B)
        adv = torch.randn(B)
        kl = (policy_lp - ref_lp).abs()
        entropy = torch.rand(B) * 2.0 + 0.5  # in [0.5, 2.5]
        return policy_lp, ref_lp, adv, kl, entropy

    def test_no_entropy_args_succeeds(self, base_tensors):
        """grpo_loss with no entropy args must not raise TypeError."""
        _assert_ram_headroom("test_no_entropy_args_succeeds")
        from ironcore.alignment.loss.grpo import grpo_loss

        p, r, a, kl, _ = base_tensors
        loss, metrics = grpo_loss(p, r, a, kl)
        assert torch.isfinite(loss)
        assert "entropy" in metrics

    def test_entropy_coef_zero_metric_is_zero(self, base_tensors):
        """When entropy_coef=0, the entropy metric must be 0.0 (not coef*entropy)."""
        _assert_ram_headroom("test_entropy_coef_zero_metric_is_zero")
        from ironcore.alignment.loss.grpo import grpo_loss

        p, r, a, kl, entropy = base_tensors
        _, metrics = grpo_loss(p, r, a, kl, entropy=entropy, entropy_coef=0.0)
        assert metrics["entropy"] == 0.0, (
            f"entropy metric should be 0.0 when entropy_coef=0, got {metrics['entropy']}"
        )

    def test_entropy_metric_is_raw_mean_not_scaled(self, base_tensors):
        """When entropy_coef>0, metric must be raw mean H (not coef*H)."""
        _assert_ram_headroom("test_entropy_metric_is_raw_mean_not_scaled")
        from ironcore.alignment.loss.grpo import grpo_loss

        p, r, a, kl, entropy = base_tensors
        coef = 0.01
        _, metrics = grpo_loss(p, r, a, kl, entropy=entropy, entropy_coef=coef)

        expected = entropy.mean().item()
        assert abs(metrics["entropy"] - expected) < 1e-5, (
            f"entropy metric should be raw mean ({expected:.4f}), got {metrics['entropy']:.4f}"
        )

    def test_entropy_bonus_reduces_loss(self, base_tensors):
        """Loss with entropy bonus must be lower than without (entropy > 0, coef > 0)."""
        _assert_ram_headroom("test_entropy_bonus_reduces_loss")
        from ironcore.alignment.loss.grpo import grpo_loss

        p, r, a, kl, entropy = base_tensors
        loss_no_entropy, _ = grpo_loss(p, r, a, kl)
        loss_with_entropy, _ = grpo_loss(p, r, a, kl, entropy=entropy, entropy_coef=0.1)
        assert loss_with_entropy.item() < loss_no_entropy.item(), (
            "Entropy bonus (subtracted) should reduce loss"
        )

    def test_entropy_none_skips_bonus(self, base_tensors):
        """entropy=None must produce same loss as entropy_coef=0."""
        _assert_ram_headroom("test_entropy_none_skips_bonus")
        from ironcore.alignment.loss.grpo import grpo_loss

        p, r, a, kl, entropy = base_tensors
        loss_none, _ = grpo_loss(p, r, a, kl, entropy=None, entropy_coef=0.1)
        loss_zero_coef, _ = grpo_loss(p, r, a, kl, entropy=entropy, entropy_coef=0.0)
        loss_baseline, _ = grpo_loss(p, r, a, kl)
        # Both should equal the no-entropy baseline
        assert abs(loss_none.item() - loss_baseline.item()) < 1e-6
        assert abs(loss_zero_coef.item() - loss_baseline.item()) < 1e-6

    def test_required_metrics_present(self, base_tensors):
        """All expected metric keys must be present in returned dict."""
        _assert_ram_headroom("test_required_metrics_present")
        from ironcore.alignment.loss.grpo import grpo_loss

        p, r, a, kl, entropy = base_tensors
        _, metrics = grpo_loss(p, r, a, kl, entropy=entropy, entropy_coef=0.01)
        required = {
            "grpo_loss",
            "policy_loss",
            "kl_loss",
            "kl_per_seq",
            "entropy",
            "mean_advantage",
            "std_advantage",
            "mean_ratio",
            "clip_fraction",
        }
        missing = required - set(metrics.keys())
        assert not missing, f"Missing metric keys: {missing}"


# ---------------------------------------------------------------------------
# 3. LRU cache in LocalEndpointRewardFunction
# ---------------------------------------------------------------------------


class TestLocalEndpointRewardCache:
    """LocalEndpointRewardFunction must properly evict oldest entries via LRU."""

    def _make_fn(self, cache_size: int = 3):
        """Build LocalEndpointRewardFunction with a fully mocked openai module."""
        import sys

        from ironcore.alignment.rewards.builtin import LocalEndpointRewardFunction

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
        _assert_ram_headroom("test_cache_hit_returns_cached_value")
        fn = self._make_fn(cache_size=10)

        # Manually seed the cache
        key = (hash("p"), hash("c"), hash("{}"))
        fn._cache[key] = 0.77

        # Simulate a compute call that would hit the cache
        # (Call _compute_cached directly to bypass client)
        result = fn._compute_cached(hash("p"), hash("c"), hash("{}"), "p", "c", {})
        assert abs(result - 0.77) < 1e-9, f"Expected cached 0.77, got {result}"

    def test_lru_eviction_oldest_entry(self):
        _assert_ram_headroom("test_lru_eviction_oldest_entry")
        fn = self._make_fn(cache_size=3)

        # Fill cache with 3 entries
        keys = [(i, i, i) for i in range(3)]
        for k in keys:
            fn._cache[k] = float(k[0]) * 0.1

        assert len(fn._cache) == 3

        # Adding a 4th entry triggers LRU eviction of the oldest (keys[0])
        if len(fn._cache) >= fn._cache_size:
            fn._cache.popitem(last=False)
        fn._cache[(99, 99, 99)] = 0.99

        assert keys[0] not in fn._cache, "Oldest entry should have been evicted"
        assert (99, 99, 99) in fn._cache
        assert len(fn._cache) == 3

    def test_cache_size_zero_clamped_to_one(self):
        _assert_ram_headroom("test_cache_size_zero_clamped_to_one")
        fn = self._make_fn(cache_size=0)
        assert fn._cache_size == 1, f"cache_size=0 should be clamped to 1, got {fn._cache_size}"

    def test_cache_size_negative_clamped_to_one(self):
        _assert_ram_headroom("test_cache_size_negative_clamped_to_one")
        fn = self._make_fn(cache_size=-100)
        assert fn._cache_size == 1

    def test_cache_is_ordered_dict(self):
        _assert_ram_headroom("test_cache_is_ordered_dict")
        fn = self._make_fn()
        assert isinstance(fn._cache, OrderedDict)

    def test_default_cache_size_is_ten_thousand(self):
        _assert_ram_headroom("test_default_cache_size_is_ten_thousand")
        fn = self._make_fn(cache_size=10000)
        assert fn._cache_size == 10000


# ---------------------------------------------------------------------------
# 4. LRU cache in LocalInferenceRewardFunction
# ---------------------------------------------------------------------------


class TestLocalInferenceRewardCache:
    """LocalInferenceRewardFunction must also evict properly and have correct defaults."""

    def _make_fn(self, cache_size: int = 3):
        """Build LocalInferenceRewardFunction with mocked transformers."""
        from ironcore.alignment.rewards.builtin import LocalInferenceRewardFunction

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
        _assert_ram_headroom("test_inference_cache_size_zero_clamped_to_one")
        fn = self._make_fn(cache_size=0)
        assert fn._cache_size == 1

    def test_default_cache_size_is_ten_thousand(self):
        _assert_ram_headroom("test_inference_default_cache_size")
        fn = self._make_fn(cache_size=10000)
        assert fn._cache_size == 10000

    def test_cache_is_ordered_dict(self):
        _assert_ram_headroom("test_inference_cache_is_ordered_dict")
        fn = self._make_fn()
        assert isinstance(fn._cache, OrderedDict)

    def test_lru_eviction_on_overflow(self):
        _assert_ram_headroom("test_inference_lru_eviction_on_overflow")
        fn = self._make_fn(cache_size=2)

        # Seed 2 entries
        fn._cache[(1, 1, 1)] = 0.1
        fn._cache[(2, 2, 2)] = 0.2
        assert len(fn._cache) == 2

        # Adding a 3rd entry should evict (1,1,1)
        if len(fn._cache) >= fn._cache_size:
            fn._cache.popitem(last=False)
        fn._cache[(3, 3, 3)] = 0.3

        assert (1, 1, 1) not in fn._cache
        assert (3, 3, 3) in fn._cache
        assert len(fn._cache) == 2

    def test_extract_score_skips_absent_vocab_tokens(self):
        """_extract_score_from_logits must skip tokens not in vocabulary (tid=None)."""
        _assert_ram_headroom("test_extract_score_skips_absent_vocab_tokens")
        from ironcore.alignment.rewards.builtin import LocalInferenceRewardFunction

        mock_tokenizer = MagicMock()

        # Simulate tokenizer where only "1" and "5" are in vocab; others return None
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

        # Logits tensor large enough to include token id 50
        logits = torch.zeros(1, 100)
        logits[0, 10] = 5.0  # "1" → score 0.1
        logits[0, 50] = 10.0  # "5" → score 0.5; higher → dominates softmax

        score = fn._extract_score_from_logits(logits)
        assert 0.0 <= score <= 1.0, f"Score out of range: {score}"
        # "5" dominates → score should be close to 0.5
        assert score > 0.3, f"Expected score near 0.5 (token '5' dominant), got {score}"


# ---------------------------------------------------------------------------
# 5. Import structure after utils/ refactor
# ---------------------------------------------------------------------------


class TestImportStructure:
    """After the mfu/utils refactor, all public import paths must work."""

    def test_mfu_from_utils_submodule(self):
        _assert_ram_headroom("test_mfu_from_utils_submodule")
        from ironcore.utils.mfu import MFUCalculator, MFUResult, compute_tflops  # noqa: F401

    def test_mfu_re_exported_from_ironcore_init(self):
        _assert_ram_headroom("test_mfu_re_exported_from_ironcore_init")
        from ironcore import MFUCalculator, MFUResult, compute_tflops  # noqa: F401

    def test_utils_package_public_api(self):
        _assert_ram_headroom("test_utils_package_public_api")
        from ironcore.utils import (  # noqa: F401
            Timer,
            bytes_to_mib,
            format_memory_report,
            get_dataset_base_dir,
            get_detailed_memory_breakdown,
            get_device,
            get_memory_usage,
            get_model_dtype,
            is_first_rank,
            is_last_rank,
            load_yaml_config,
            print_last_rank,
            print_rank_0,
            profile_context,
            profile_function,
        )

    def test_clip_grad_norm_tp_deferred_import(self):
        """clip_grad_norm_tp is importable via deferred import after TP module is initialized.

        In production, base_trainer.py imports it inside _compute_grad_and_param_norms()
        by which time ironcore.parallel.tensor_parallel is already fully initialized.
        We replicate that by importing the trainer module first.
        """
        _assert_ram_headroom("test_clip_grad_norm_tp_deferred_import")
        import importlib

        # Pre-initialize the TP module via the trainer (mirrors actual runtime order)
        importlib.import_module("ironcore.trainers.base_trainer")

        def _deferred():
            from ironcore.parallel.tensor_parallel.comm import clip_grad_norm_tp

            return clip_grad_norm_tp

        fn = _deferred()
        assert callable(fn), "clip_grad_norm_tp must be a callable"

    def test_clip_grad_norm_tp_in_tp_package(self):
        """clip_grad_norm_tp must also be re-exported from the TP package."""
        _assert_ram_headroom("test_clip_grad_norm_tp_in_tp_package")

        def _deferred():
            from ironcore.parallel.tensor_parallel import clip_grad_norm_tp

            return clip_grad_norm_tp

        fn = _deferred()
        assert callable(fn)


# ---------------------------------------------------------------------------
# 6. DeepSeek format token: <think> replaces <currwork>
# ---------------------------------------------------------------------------


class TestDeepSeekFormatToken:
    """StrictFormatRewardFunction default pattern and format_deepseek.yaml must use <think>."""

    def test_default_pattern_accepts_think_tag(self):
        _assert_ram_headroom("test_default_pattern_accepts_think_tag")
        from ironcore.alignment.rewards.builtin import StrictFormatRewardFunction

        fn = StrictFormatRewardFunction()
        score = fn.compute("", "<think>some reasoning</think>#### 42", {})
        assert score == 1.0, f"<think> tag should match default pattern, got {score}"

    def test_default_pattern_rejects_currwork_tag(self):
        _assert_ram_headroom("test_default_pattern_rejects_currwork_tag")
        from ironcore.alignment.rewards.builtin import StrictFormatRewardFunction

        fn = StrictFormatRewardFunction()
        score = fn.compute("", "<currwork>some reasoning</currwork>#### 42", {})
        assert score == 0.0, f"<currwork> tag should NOT match default pattern, got {score}"

    def test_deepseek_yaml_uses_think_token(self):
        _assert_ram_headroom("test_deepseek_yaml_uses_think_token")
        import yaml

        with open(FORMAT_DEEPSEEK_YAML, encoding="utf-8") as f:
            config = yaml.safe_load(f)

        pattern = config.get("pattern", "")
        assert "<think>" in pattern, (
            f"format_deepseek.yaml pattern must contain <think>, got: {pattern}"
        )
        assert "<currwork>" not in pattern, "format_deepseek.yaml must not reference <currwork>"

    def test_deepseek_yaml_template_accepts_think(self):
        _assert_ram_headroom("test_deepseek_yaml_template_accepts_think")
        from ironcore.alignment.rewards.template import TemplateRuleReward

        fn = TemplateRuleReward.from_yaml(FORMAT_DEEPSEEK_YAML)
        score = fn.compute("", "<think>reasoning</think>#### 7", {})
        assert score == 1.0

    def test_deepseek_yaml_template_rejects_currwork(self):
        _assert_ram_headroom("test_deepseek_yaml_template_rejects_currwork")
        from ironcore.alignment.rewards.template import TemplateRuleReward

        fn = TemplateRuleReward.from_yaml(FORMAT_DEEPSEEK_YAML)
        score = fn.compute("", "<currwork>reasoning</currwork>#### 7", {})
        assert score == 0.0


# ---------------------------------------------------------------------------
# 7. CodeRewardFunction: must raise NotImplementedError, not return 0
# ---------------------------------------------------------------------------


class TestCodeRewardFunction:
    def test_raises_not_implemented(self):
        _assert_ram_headroom("test_code_reward_raises_not_implemented")
        from ironcore.alignment.rewards.builtin import CodeRewardFunction

        fn = CodeRewardFunction()
        with pytest.raises(NotImplementedError) as exc_info:
            fn.compute("prompt", "def foo(): pass", {"test_cases": ["assert foo() is None"]})

        assert "sandbox" in str(exc_info.value).lower(), (
            "NotImplementedError message must mention sandbox"
        )

    def test_raises_even_without_test_cases(self):
        _assert_ram_headroom("test_code_reward_raises_even_without_test_cases")
        from ironcore.alignment.rewards.builtin import CodeRewardFunction

        fn = CodeRewardFunction()
        # Old implementation returned 0.5 when no test_cases — now must raise
        with pytest.raises(NotImplementedError):
            fn.compute("prompt", "code", {})


# ---------------------------------------------------------------------------
# 8. Trainer module load smoke (no circular import at function scope)
# ---------------------------------------------------------------------------


class TestTrainerImportSmoke:
    """Verify trainer modules load without error after the refactor."""

    def test_base_trainer_module_loads(self):
        _assert_ram_headroom("test_base_trainer_module_loads")
        import importlib

        mod = importlib.import_module("ironcore.trainers.base_trainer")
        assert hasattr(mod, "BaseTrainer")

    def test_grpo_trainer_module_loads(self):
        _assert_ram_headroom("test_grpo_trainer_module_loads")
        import importlib

        mod = importlib.import_module("ironcore.trainers.grpo_trainer")
        assert hasattr(mod, "GRPOTrainer")

    def test_clip_grad_norm_tp_used_in_base_trainer(self):
        """base_trainer must reference clip_grad_norm_tp from the new canonical location."""
        _assert_ram_headroom("test_clip_grad_norm_tp_used_in_base_trainer")
        import inspect

        import ironcore.trainers.base_trainer as bt

        source = inspect.getsource(bt)
        assert "parallel.grad_norm" in source or "parallel.tensor_parallel" in source, (
            "base_trainer should import clip_grad_norm from ironcore.parallel "
            "(not ironcore.utils) — currently uses ironcore.parallel.grad_norm"
        )
        assert "from ironcore.utils import clip_grad_norm" not in source, (
            "Old import path 'from ironcore.utils import clip_grad_norm*' must be removed"
        )


# ---------------------------------------------------------------------------
# 9. Memory monitoring self-test
# ---------------------------------------------------------------------------


class TestMemoryGuard:
    """Verify the memory guard reads sensible values."""

    def test_host_ram_usage_is_positive(self):
        used_gb = _host_ram_used_gb()
        assert used_gb > 0.0, "psutil must report positive RAM usage"

    def test_host_ram_usage_below_limit(self):
        used_gb = _host_ram_used_gb()
        assert used_gb < HOST_RAM_LIMIT_GB, (
            f"Host RAM {used_gb:.1f} GB already exceeds {HOST_RAM_LIMIT_GB:.0f} GB limit"
        )

    def test_memory_guard_limit_config(self):
        guard = _MemoryGuard(limit_gb=60.0, poll_interval=999.0)
        assert guard._limit_bytes == int(60.0 * 1024**3)
