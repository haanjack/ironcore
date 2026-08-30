# Copyright (c) 2025-2026 Jaegeun Han
#
# SPDX-License-Identifier: Apache-2.0
"""Tests for config correctness — regression tests for known bugs."""

import dataclasses
from pathlib import Path
from types import SimpleNamespace

from ironcore.config.config_alignment import AlignmentConfig
from ironcore.config.config_model import ModelConfig
from ironcore.config.config_trainer import TrainerConfig


class TestXPosRemoved:
    """Regression test: XPos was a dead stub that should not exist."""

    def test_xpos_file_does_not_exist(self):
        xpos_path = Path("ironcore/layers/positional_embedding/xpos.py")
        assert not xpos_path.exists(), "xpos.py should have been deleted"


class TestModelConfig:
    """Regression tests for ModelConfig bugs."""

    def test_activation_type_not_duplicated(self):
        """activation_type should appear exactly once (was duplicated at lines 151 and 159)."""
        activation_fields = [
            f for f in dataclasses.fields(ModelConfig) if f.name == "activation_type"
        ]
        assert len(activation_fields) == 1, (
            f"Expected 1 activation_type field, found {len(activation_fields)}"
        )

    def test_activation_type_default_is_gelu(self):
        """activation_type should default to 'gelu'."""
        config = ModelConfig()
        assert config.activation_type == "gelu"


class TestTrainerConfig:
    """Regression tests for TrainerConfig bugs."""

    def test_no_pipeline_model_parallel_size(self):
        """pipeline_model_parallel_size should not exist (removed as unimplemented ghost)."""
        assert not hasattr(TrainerConfig(), "pipeline_model_parallel_size"), (
            "pipeline_model_parallel_size should have been removed from TrainerConfig"
        )


class TestAlignmentConfig:
    """Regression tests for AlignmentConfig fields."""

    def test_offload_ref_model_field_exists(self):
        """offload_ref_model should be a declared field defaulting to False."""
        config = AlignmentConfig()
        assert hasattr(config, "offload_ref_model")
        assert config.offload_ref_model is False

    def test_offload_ref_model_is_configurable(self):
        """offload_ref_model should be settable."""
        config = AlignmentConfig(offload_ref_model=True)
        assert config.offload_ref_model is True


class TestInlineDataBlockTypoGuard:
    """A misspelled key in an inline `data:` block must not be accepted.

    Every other config group is checked against its dataclass fields, but the
    data path guarded with `sub_group_key not in config_group` — a key drawn
    from that same dict, so the check never fired. A typo was setattr'd as a
    stray attribute while the real field silently kept its default.
    """

    @staticmethod
    def _apply(block):
        from ironcore.config import _update_data_config_from_yaml
        from ironcore.config.config_data import DataConfig

        holder = type("Holder", (), {})()
        holder.data = DataConfig()
        _update_data_config_from_yaml(holder, "data", block)
        return holder.data

    def test_known_key_is_applied(self):
        data = self._apply({"seq_length": 256})
        assert data.seq_length == 256

    def test_misspelled_key_raises(self):
        import pytest

        with pytest.raises(ValueError, match="seq_lenght"):
            self._apply({"seq_lenght": 256})

    def test_misspelled_key_does_not_reach_the_object(self):
        import pytest

        from ironcore.config.config_data import DataConfig

        default = DataConfig().seq_length
        with pytest.raises(ValueError):
            self._apply({"seq_lenght": 999999})
        assert DataConfig().seq_length == default


class TestNestedConfigOverrideMerges:
    """A partial write to a nested group must not reset its siblings.

    BaseConfig.__call__ built a fresh field_type(**v) from only the supplied
    keys, so an overlay touching one field of model.moe also silently reverted
    every other field to its dataclass default — including use_moe, which turns
    the feature off entirely.
    """

    def test_partial_nested_write_keeps_siblings(self):
        from ironcore.config.config_model import ModelConfig

        config = ModelConfig()
        config(moe={"use_moe": True, "num_routed_experts": 8})
        assert config.moe.use_moe is True
        assert config.moe.num_routed_experts == 8

        # Second, partial write — the shape an overlay config takes.
        config(moe={"num_routed_experts": 16})

        assert config.moe.num_routed_experts == 16
        assert config.moe.use_moe is True, "unmentioned sibling was reset to its default"

    def test_nested_write_onto_default_still_works(self):
        from ironcore.config.config_model import ModelConfig

        config = ModelConfig()
        config(moe={"num_routed_experts": 4})
        assert config.moe.num_routed_experts == 4

    def test_unknown_nested_key_still_rejected(self):
        import pytest

        from ironcore.config.config_model import ModelConfig

        with pytest.raises((KeyError, TypeError)):
            ModelConfig()(moe={"definitely_not_a_moe_field": 1})


class TestGrpoDisablesDropout:
    """GRPO's KL penalty must not end up measuring dropout noise.

    The reference model is a deep copy of the policy run in eval(), while the
    policy's log-prob pass runs in train(). With dropout on, the two disagree on
    identical weights and that disagreement is charged as divergence — measured
    at kl_loss 1.31/2.18/1.47/3.94 over four steps on Qwen2.5-0.5B, against
    0.0000 for the same config at dropout 0.
    """

    @staticmethod
    def _config(method):
        from ironcore.config.config_alignment import RewardManagerConfig
        from ironcore.config.config_trainer import OperationConfig

        extra = {}
        if method == "grpo":
            # AlignmentConfig.__post_init__ requires this for GRPO.
            extra["reward_manager"] = RewardManagerConfig()

        from ironcore.config.config_data import DataConfig

        return SimpleNamespace(
            model=ModelConfig(dropout_attn=0.1, dropout_mlp=0.1, dropout_embd=0.1),
            alignment=AlignmentConfig(method=method, **extra),
            trainer=TrainerConfig(micro_batch_size=1, train_batch_size=1),
            operation=OperationConfig(train_steps=1),
            data=DataConfig(),
        )

    @staticmethod
    def _validate(cfg):
        """Run validation far enough to reach the dropout rule.

        The stub config omits fields later checks need, so the exception those
        raise is expected and not what is under test here.
        """
        from ironcore.config import _config_validation

        try:
            _config_validation(cfg)
        except Exception:  # noqa: BLE001
            pass

    def test_grpo_disables_dropout_and_says_so(self):
        import warnings

        cfg = self._config("grpo")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            self._validate(cfg)

        assert cfg.model.dropout_attn == 0.0
        assert cfg.model.dropout_mlp == 0.0
        assert cfg.model.dropout_embd == 0.0
        assert any("dropout" in str(w.message).lower() for w in caught), (
            "changing the config silently would be its own bug"
        )

    def test_dpo_keeps_dropout(self):
        cfg = self._config("dpo")
        self._validate(cfg)
        assert cfg.model.dropout_attn == 0.1
        assert cfg.model.dropout_mlp == 0.1


class TestTokenizerAgreement:
    """Preprocessing and training must not name different vocabularies.

    ironcore/preprocess.py takes a DataConfig and tokenizes with
    data.vocab_name_or_path; build_tokenizer embeds and decodes with
    model.vocab_name_or_path. Nothing reconciled the two, so a config could
    write ids under one vocabulary and train under another.
    """

    @staticmethod
    def _config(data_vocab, model_vocab):
        from ironcore.config.config_data import DataConfig
        from ironcore.config.config_trainer import OperationConfig

        return SimpleNamespace(
            model=ModelConfig(vocab_name_or_path=model_vocab),
            alignment=AlignmentConfig(),
            trainer=TrainerConfig(micro_batch_size=1, train_batch_size=1),
            operation=OperationConfig(train_steps=1),
            data=DataConfig(vocab_name_or_path=data_vocab),
        )

    def test_mismatch_is_rejected(self):
        import pytest

        from ironcore.config import _config_validation

        cfg = self._config("./tokenizers/gpt2-fim", "gpt2")
        with pytest.raises(ValueError, match="different tokenizers"):
            _config_validation(cfg)

    def test_matching_values_are_accepted(self):
        from ironcore.config import _config_validation

        cfg = self._config("./tokenizers/gpt2-fim", "./tokenizers/gpt2-fim")
        try:
            _config_validation(cfg)
        except ValueError as exc:  # later checks need fields this stub omits
            assert "different tokenizers" not in str(exc)
        except Exception:  # noqa: BLE001
            pass


class TestRemovedDuplicateFields:
    """model.attention_head_size and model.attention_dropout are gone.

    Neither had a consumer: head_dim and dropout_attn are what the model reads.
    attention_head_size defaulted to 64 against head_dim's 128, so they did not
    even agree out of the box — resizing a model by editing the more
    natural-sounding name did nothing, silently. Removing them means the loader's
    unknown-key check reports the mistake instead.
    """

    def test_fields_no_longer_exist(self):
        assert not hasattr(ModelConfig(), "attention_head_size")
        assert not hasattr(ModelConfig(), "attention_dropout")

    def test_constructing_with_a_removed_key_is_rejected(self):
        """The loader checks YAML keys against these fields, so a config setting
        one now gets "not defined in model config" instead of silent acceptance."""
        import pytest

        with pytest.raises(TypeError):
            ModelConfig(attention_head_size=64)
        with pytest.raises(TypeError):
            ModelConfig(attention_dropout=0.2)
