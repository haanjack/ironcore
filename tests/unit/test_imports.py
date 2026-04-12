# Copyright (c) 2025-2026 Jaegeun Han
#
# SPDX-License-Identifier: Apache-2.0

"""Import and module-load smoke tests."""

import importlib
import inspect


class TestMFUImports:
    """MFU public API import paths."""

    def test_mfu_from_utils_submodule(self):
        from ironcore.utils.mfu import MFUCalculator, MFUResult, compute_tflops  # noqa: F401

    def test_mfu_re_exported_from_ironcore_init(self):
        from ironcore import MFUCalculator, MFUResult, compute_tflops  # noqa: F401

    def test_utils_package_public_api(self):
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


class TestParallelImports:
    """Parallel module import paths."""

    def test_clip_grad_norm_tp_deferred_import(self):
        """clip_grad_norm_tp is importable after TP module is initialized."""
        importlib.import_module("ironcore.trainers.base_trainer")

        def _deferred():
            from ironcore.parallel.tensor_parallel.comm import clip_grad_norm_tp

            return clip_grad_norm_tp

        fn = _deferred()
        assert callable(fn), "clip_grad_norm_tp must be a callable"

    def test_clip_grad_norm_tp_in_tp_package(self):
        """clip_grad_norm_tp must be re-exported from the TP package."""

        def _deferred():
            from ironcore.parallel.tensor_parallel import clip_grad_norm_tp

            return clip_grad_norm_tp

        fn = _deferred()
        assert callable(fn)


class TestTrainerModuleLoad:
    """Trainer modules load without circular import errors."""

    def test_base_trainer_module_loads(self):
        mod = importlib.import_module("ironcore.trainers.base_trainer")
        assert hasattr(mod, "BaseTrainer")

    def test_grpo_trainer_module_loads(self):
        mod = importlib.import_module("ironcore.trainers.grpo_trainer")
        assert hasattr(mod, "GRPOTrainer")

    def test_clip_grad_norm_canonical_import(self):
        """base_trainer must import clip_grad_norm from ironcore.parallel, not ironcore.utils."""
        import ironcore.trainers.base_trainer as bt

        source = inspect.getsource(bt)
        assert "parallel.grad_norm" in source or "parallel.tensor_parallel" in source, (
            "base_trainer should import clip_grad_norm from ironcore.parallel"
        )
        assert "from ironcore.utils import clip_grad_norm" not in source, (
            "Old import path 'from ironcore.utils import clip_grad_norm*' must be removed"
        )
