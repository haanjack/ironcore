# Copyright (c) 2025-2026 Jaegeun Han
#
# SPDX-License-Identifier: Apache-2.0

"""
Unit test for get_evaluators()'s handling of unknown task names.

Regression coverage: an unknown/misspelled task name used to be silently
skipped (print() + continue), so a config typo produced no eval output and
no error. Now raises ValueError so the misconfiguration is surfaced.
"""

import logging

import pytest

from ironcore.eval.evaluator import get_evaluators


class _FakeTokenizer:
    eos_token = "<eos>"


def test_unknown_task_name_raises(monkeypatch):
    monkeypatch.setattr("ironcore.eval.evaluator.get_tokenizer", _FakeTokenizer)
    monkeypatch.setattr(
        "ironcore.eval.evaluator.get_logger", lambda: logging.getLogger("test_evaluator")
    )

    with pytest.raises(ValueError, match="could not be loaded"):
        get_evaluators([{"name": "this_task_does_not_exist"}])
