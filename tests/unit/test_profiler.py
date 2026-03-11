# Copyright (c) 2025-2026 Jaegeun Han
#
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for profiling components.

Covers test plan items:
  Phase 1 : B2, B3, B5, B10
  Phase 2 : B8, B9
  Phase 3 : F1, F2, F6
  Phase 4 : F4, F5
  Regression: singleton isolation, re-initialization reset
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from unittest import mock

import pytest

from ironcore.config import ProfilerConfig
from ironcore.profiler import (
    CommProfiler,
    LayerTimingCollector,
    ProfileManager,
    TimedDataIterator,
    get_layer_timing_collector,
    timed_comm,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _reset_singletons():
    """Reset singleton instances so each test starts from a clean state."""
    CommProfiler._instance = None
    LayerTimingCollector._instance = None


def _make_profiler_config(tmp_path: Path, **kwargs):
    """Return a MainConfig with ProfilerConfig wired to *tmp_path*."""
    # import here so test module collection works without heavy deps loaded
    from tests.fixtures.config_fixtures import create_small_test_config

    config = create_small_test_config()
    config.profiler = ProfilerConfig(
        output_dir=str(tmp_path),
        **kwargs,
    )
    return config


def _make_profile_manager(tmp_path: Path, **profiler_kwargs) -> ProfileManager:
    """Build a ProfileManager with a mocked logger to avoid global-states dependency."""
    config = _make_profiler_config(tmp_path, **profiler_kwargs)
    with mock.patch("ironcore.profiler.get_logger", return_value=mock.MagicMock()):
        pm = ProfileManager(config)
    return pm


# ---------------------------------------------------------------------------
# CommProfiler
# ---------------------------------------------------------------------------


class TestCommProfiler:
    def setup_method(self):
        _reset_singletons()

    # B-reg: singleton identity
    def test_singleton_returns_same_instance(self):
        a = CommProfiler()
        b = CommProfiler()
        assert a is b

    def test_disabled_by_default(self):
        p = CommProfiler()
        assert not p.enabled

    def test_record_no_op_when_disabled(self):
        p = CommProfiler()
        p.record("op", 10.0)
        # manually enable to inspect — stats must still be empty
        p.enabled = True
        stats = p.get_and_reset_stats()
        assert stats == {}

    def test_enable_resets_existing_stats(self):
        p = CommProfiler()
        p.enable()
        p.record("op", 5.0)
        # re-enabling must reset
        p.enable()
        stats = p.get_and_reset_stats()
        assert stats == {}

    def test_record_accumulates_multiple_ops(self):
        p = CommProfiler()
        p.enable()
        p.record("tp_all_reduce", 2.0)
        p.record("tp_all_reduce", 4.0)
        p.record("ep_all_to_all_tokens", 1.5)
        stats = p.get_and_reset_stats()

        assert set(stats.keys()) == {"tp_all_reduce", "ep_all_to_all_tokens"}
        s = stats["tp_all_reduce"]
        assert s["count"] == 2
        assert s["total_ms"] == pytest.approx(6.0)
        assert s["mean_ms"] == pytest.approx(3.0)
        assert s["max_ms"] == pytest.approx(4.0)

    def test_get_and_reset_clears_stats(self):
        p = CommProfiler()
        p.enable()
        p.record("op", 1.0)
        p.get_and_reset_stats()
        assert p.get_and_reset_stats() == {}

    def test_disable_stops_recording(self):
        p = CommProfiler()
        p.enable()
        p.record("op", 1.0)
        p.get_and_reset_stats()  # flush the pre-disable record
        p.disable()
        p.record("op", 99.0)  # must not be stored (disabled)
        # peek without enabling (which would reset)
        p.enabled = True
        stats = p.get_and_reset_stats()
        # 99.0 was recorded while disabled → must not appear
        assert stats == {}


# ---------------------------------------------------------------------------
# timed_comm context manager  (F1 — zero overhead when disabled)
# ---------------------------------------------------------------------------


class TestTimedComm:
    def setup_method(self):
        _reset_singletons()

    def test_records_duration_when_enabled(self):
        p = CommProfiler()
        p.enable()
        with timed_comm("test_op"):
            time.sleep(0.005)
        stats = p.get_and_reset_stats()
        assert "test_op" in stats
        assert stats["test_op"]["count"] == 1
        assert stats["test_op"]["total_ms"] >= 1.0  # at least 1 ms

    def test_no_overhead_when_disabled(self):
        # CommProfiler disabled by default
        with timed_comm("test_op"):
            pass
        p = CommProfiler()
        p.enabled = True  # peek without enabling (which resets)
        assert p.get_and_reset_stats() == {}

    def test_exception_inside_context_still_records(self):
        p = CommProfiler()
        p.enable()
        with pytest.raises(ValueError):
            with timed_comm("bad_op"):
                raise ValueError("oops")
        stats = p.get_and_reset_stats()
        assert "bad_op" in stats


# ---------------------------------------------------------------------------
# LayerTimingCollector
# ---------------------------------------------------------------------------


class TestLayerTimingCollector:
    def setup_method(self):
        _reset_singletons()

    def test_singleton_returns_same_instance(self):
        a = LayerTimingCollector()
        b = LayerTimingCollector()
        assert a is b

    def test_get_layer_timing_collector_is_singleton(self):
        a = get_layer_timing_collector()
        b = get_layer_timing_collector()
        assert a is b

    def test_disabled_by_default(self):
        c = LayerTimingCollector()
        assert not c.enabled

    def test_summary_when_no_data_collected(self):
        c = LayerTimingCollector()
        c.enable()
        assert "No layer timing data" in c.get_summary()

    def test_reset_clears_pending_and_completed(self):
        c = LayerTimingCollector()
        # inject fake state
        c._pending[1] = ("Layer", "forward", None)
        c._completed.append(("Layer", "forward", None, None))
        c.reset()
        assert c._pending == {}
        assert c._completed == []

    def test_start_end_no_op_without_cuda(self):
        """Without CUDA, start/end must not raise and must not populate state."""
        c = LayerTimingCollector()
        c.enable()
        with mock.patch("torch.cuda.is_available", return_value=False):
            c.start(42, "TestLayer", "forward")
            c.end(42)
        assert c._pending == {}
        assert c._completed == []

    # F6 — recompute overwrite logged
    def test_overwrite_pending_entry_logs_debug(self, caplog):
        """Calling start() twice for the same module_id logs a debug message."""
        c = LayerTimingCollector()
        c.enable()

        fake_event = mock.MagicMock()
        with (
            mock.patch("torch.cuda.is_available", return_value=True),
            mock.patch("torch.cuda.Event", return_value=fake_event),
        ):
            with caplog.at_level(logging.DEBUG, logger="ironcore.profiler"):
                c.start(99, "TransformerLayer", "forward")
                c.start(99, "TransformerLayer", "forward")  # overwrite

        assert any("overwriting pending entry" in r.message for r in caplog.records)

    # F6 — summary table format
    def test_summary_table_format(self):
        """get_summary() must produce a table with header lines and layer rows."""
        c = LayerTimingCollector()
        c.enable()

        start_ev = mock.MagicMock()
        end_ev = mock.MagicMock()
        start_ev.elapsed_time.return_value = 5.0  # 5 ms

        c._completed.append(("TransformerLayer", "forward", start_ev, end_ev))
        c._completed.append(("TransformerLayer", "backward", start_ev, end_ev))

        with (
            mock.patch("torch.cuda.is_available", return_value=True),
            mock.patch("torch.cuda.synchronize"),
        ):
            summary = c.get_summary()

        assert "=" * 10 in summary
        assert "Fwd (ms)" in summary
        assert "TransformerLayer" in summary


# ---------------------------------------------------------------------------
# TimedDataIterator  (F5)
# ---------------------------------------------------------------------------


class TestTimedDataIterator:
    def test_iterates_correctly(self):
        timed = TimedDataIterator(iter([10, 20, 30]))
        assert [next(timed) for _ in range(3)] == [10, 20, 30]

    def test_measures_elapsed_time(self):
        def slow_gen():
            for _ in range(2):
                time.sleep(0.01)
                yield 1

        timed = TimedDataIterator(slow_gen())
        next(timed)
        stats = timed.get_and_reset_stats()
        assert stats["count"] == 1
        assert stats["total_ms"] >= 5.0  # at least 5 ms

    def test_get_and_reset_returns_average_info(self):
        timed = TimedDataIterator(iter([1, 2, 3]))
        next(timed)
        next(timed)
        stats = timed.get_and_reset_stats()
        assert stats["count"] == 2
        assert stats["total_ms"] >= 0.0

    def test_get_and_reset_clears_state(self):
        timed = TimedDataIterator(iter([1, 2]))
        next(timed)
        timed.get_and_reset_stats()
        stats = timed.get_and_reset_stats()
        assert stats["count"] == 0
        assert stats["total_ms"] == 0.0

    # B5 / F5 — len propagates
    def test_len_propagates_from_sized_iterator(self):
        timed = TimedDataIterator(range(7))
        assert len(timed) == 7

    # F5 — unsized iterator raises TypeError
    def test_len_raises_type_error_for_unsized_iterator(self):
        timed = TimedDataIterator(iter([1, 2, 3]))
        with pytest.raises(TypeError):
            len(timed)

    def test_iter_returns_self(self):
        timed = TimedDataIterator(iter([]))
        assert iter(timed) is timed


# ---------------------------------------------------------------------------
# ProfileManager
# ---------------------------------------------------------------------------


class TestProfileManager:
    def setup_method(self):
        _reset_singletons()

    # B1 — is_active starts False
    def test_is_active_starts_false(self, tmp_path):
        pm = _make_profile_manager(tmp_path)
        assert pm.is_active is False

    # B1 — manual start activates
    def test_start_sets_is_active(self, tmp_path):
        pm = _make_profile_manager(tmp_path)
        pm.start()
        assert pm.is_active is True

    # B1 — stop deactivates
    def test_stop_clears_is_active(self, tmp_path):
        pm = _make_profile_manager(tmp_path)
        pm.start()
        pm.stop()
        assert pm.is_active is False

    # B1 — step() triggers start at configured step
    def test_step_triggers_start_at_configured_step(self, tmp_path):
        pm = _make_profile_manager(tmp_path, start=2, end=10)
        pm.step(1)
        assert pm.is_active is False
        pm.step(2)
        assert pm.is_active is True

    # B1 — step() triggers stop at configured end
    def test_step_triggers_stop_at_configured_end(self, tmp_path):
        pm = _make_profile_manager(tmp_path, start=2, end=4)
        pm.step(2)
        assert pm.is_active is True
        pm.step(4)
        assert pm.is_active is False

    # B1 — active between start and end, inactive before and after
    def test_active_window_is_correct(self, tmp_path):
        pm = _make_profile_manager(tmp_path, start=3, end=5)
        for step in [1, 2]:
            pm.step(step)
            assert pm.is_active is False
        pm.step(3)
        assert pm.is_active is True
        pm.step(4)
        assert pm.is_active is True
        pm.step(5)
        assert pm.is_active is False

    # B9 — CUDA synchronize called in start()
    def test_cuda_synchronize_called_before_capture(self, tmp_path):
        pm = _make_profile_manager(tmp_path)
        with (
            mock.patch("torch.cuda.is_available", return_value=True),
            mock.patch("torch.cuda.synchronize") as mock_sync,
        ):
            pm.start()
        mock_sync.assert_called_once()

    # Wrong rank should not profile
    def test_wrong_rank_does_not_profile(self, tmp_path):
        pm = _make_profile_manager(tmp_path, ranks=[1])  # rank 0 excluded
        assert pm.should_profile is False
        pm.start()
        assert pm.is_active is False

    # step() is a no-op for non-profiling ranks
    def test_step_no_op_for_non_profiling_rank(self, tmp_path):
        pm = _make_profile_manager(tmp_path, ranks=[1], start=1, end=5)
        pm.step(1)
        assert pm.is_active is False

    # Regression — singleton reset on re-initialization
    def test_singleton_state_reset_on_reinit(self, tmp_path):
        """Second ProfileManager on the same process must start with empty comm stats."""
        pm1 = _make_profile_manager(tmp_path, comm_profiler=False)
        # Force-enable comm profiler and record fake data to dirty the singleton
        pm1._comm_profiler.enable()
        pm1._comm_profiler.record("old_op", 999.9)

        # Re-initialize from a fresh CommProfiler (simulate second PM creation)
        _reset_singletons()
        pm2 = _make_profile_manager(tmp_path, comm_profiler=False)
        stats = pm2._comm_profiler.get_and_reset_stats()
        assert stats == {}

    # F1 — comm profiler enabled/disabled by ProfileManager
    def test_comm_profiler_enabled_on_start(self, tmp_path):
        pm = _make_profile_manager(tmp_path, comm_profiler=True)
        pm.start()
        assert pm._comm_profiler.enabled is True

    def test_comm_profiler_disabled_on_stop(self, tmp_path):
        pm = _make_profile_manager(tmp_path, comm_profiler=True)
        pm.start()
        pm.stop()
        assert pm._comm_profiler.enabled is False

    # F1 — per-step comm stats logged at debug level
    def test_per_step_comm_stats_logged(self, tmp_path):
        pm = _make_profile_manager(tmp_path, comm_profiler=True, start=1, end=99)
        pm.step(1)  # activates profiler
        assert pm.is_active is True

        # inject a fake stat directly
        pm._comm_profiler._stats["tp_all_reduce"] = [2.5, 3.5]
        pm.step(2)  # should log and reset
        pm.logger.debug.assert_called()  # type: ignore[attr-defined]

    # F2 — memory snapshot path constructed correctly
    def test_memory_snapshot_path_construction(self, tmp_path):
        pm = _make_profile_manager(tmp_path, memory_snapshot=True, name="test_run")
        expected_path = tmp_path / f"test_run_{pm.current_version}_rank{pm.rank}_memory.pickle"

        calls = []

        def fake_dump(path):
            calls.append(path)

        with (
            mock.patch("torch.cuda.is_available", return_value=True),
            mock.patch("torch.cuda.memory._record_memory_history"),
            mock.patch("torch.cuda.memory._dump_snapshot", side_effect=fake_dump),
        ):
            pm.start()
            pm.stop()

        assert len(calls) == 1
        assert calls[0] == str(expected_path)

    # F2 — graceful failure when CUDA unavailable
    def test_memory_snapshot_skipped_without_cuda(self, tmp_path):
        pm = _make_profile_manager(tmp_path, memory_snapshot=True)
        with mock.patch("torch.cuda.is_available", return_value=False):
            pm.start()
            pm.stop()  # must not raise
        assert pm.is_active is False

    # F2 — graceful failure if dump raises
    def test_memory_snapshot_failure_does_not_crash(self, tmp_path):
        pm = _make_profile_manager(tmp_path, memory_snapshot=True)
        with (
            mock.patch("torch.cuda.is_available", return_value=True),
            mock.patch("torch.cuda.memory._record_memory_history"),
            mock.patch("torch.cuda.memory._dump_snapshot", side_effect=RuntimeError("disk full")),
        ):
            pm.start()
            pm.stop()  # warning logged, no exception propagated
        assert pm.is_active is False

    # F6 — layer timing enabled on start, disabled on stop
    def test_layer_timing_enabled_on_start(self, tmp_path):
        pm = _make_profile_manager(tmp_path, layer_timing=True)
        pm.start()
        assert pm._layer_timing.enabled is True

    def test_layer_timing_disabled_on_stop(self, tmp_path):
        pm = _make_profile_manager(tmp_path, layer_timing=True)
        pm.start()
        pm.stop()
        assert pm._layer_timing.enabled is False

    # F6 — summary logged at stop
    def test_layer_timing_summary_logged_on_stop(self, tmp_path):
        pm = _make_profile_manager(tmp_path, layer_timing=True)
        pm.start()
        pm.stop()
        # logger.info should have been called with the summary
        pm.logger.info.assert_called()  # type: ignore[attr-defined]

    # F5 — wrap_data_iterator returns TimedDataIterator
    def test_wrap_data_iterator_returns_timed(self, tmp_path):
        pm = _make_profile_manager(tmp_path)
        raw = iter([1, 2, 3])
        timed = pm.wrap_data_iterator(raw)
        assert isinstance(timed, TimedDataIterator)

    # F5 — get_data_load_stats returns None when no iterator wrapped
    def test_get_data_load_stats_none_without_wrapper(self, tmp_path):
        pm = _make_profile_manager(tmp_path)
        assert pm.get_data_load_stats() is None

    # F5 — get_data_load_stats returns stats after next() calls
    def test_get_data_load_stats_after_iteration(self, tmp_path):
        pm = _make_profile_manager(tmp_path)
        pm.wrap_data_iterator(iter([1, 2, 3]))
        next(pm._timed_data_iter)
        next(pm._timed_data_iter)
        stats = pm.get_data_load_stats()
        assert stats is not None
        assert stats["count"] == 2

    # F4 — chrome trace mutual exclusion: on_trace_ready must be None
    def test_chrome_trace_makes_on_trace_ready_none(self, tmp_path):
        # patch the name 'profile' as imported into the profiler module
        with (
            mock.patch("ironcore.profiler.get_logger", return_value=mock.MagicMock()),
            mock.patch("ironcore.profiler.profile") as mock_profile_cls,
        ):
            mock_profile_cls.return_value = mock.MagicMock()
            config = _make_profiler_config(
                tmp_path,
                torch_profiler=True,
                export_chrome_trace=True,
            )
            ProfileManager(config)  # calls _init_profilers internally

        # Inspect the kwargs profile() was called with
        assert mock_profile_cls.called, "torch.profiler.profile was not instantiated"
        call_kwargs = mock_profile_cls.call_args.kwargs
        assert call_kwargs.get("on_trace_ready") is None

    # B2 / F4 — torch profiler can be initialized without AttributeError
    def test_torch_profiler_init_no_attribute_error(self, tmp_path):
        with (
            mock.patch("ironcore.profiler.get_logger", return_value=mock.MagicMock()),
            mock.patch("torch.profiler.profile") as mock_profile_cls,
            mock.patch("torch.profiler.tensorboard_trace_handler", return_value=None),
        ):
            mock_profile_cls.return_value = mock.MagicMock()
            # Should not raise AttributeError
            pm = _make_profile_manager(
                tmp_path,
                torch_profiler=True,
                export_chrome_trace=False,
            )
        assert pm.torch_profiler is not None

    # Version auto-increment
    def test_version_starts_at_v0_on_empty_dir(self, tmp_path):
        pm = _make_profile_manager(tmp_path, name="myrun")
        assert pm.current_version == "v0"

    def test_version_increments_when_previous_exists(self, tmp_path):
        (tmp_path / "myrun_v0.json").touch()
        (tmp_path / "myrun_v1.json").touch()
        pm = _make_profile_manager(tmp_path, name="myrun")
        assert pm.current_version == "v2"


# ---------------------------------------------------------------------------
# BaseModule.register_profile_hooks  (B3, B8, B10)
# ---------------------------------------------------------------------------


class TestBaseModuleHooks:
    def _make_config(self):
        from tests.fixtures.config_fixtures import create_small_test_config

        return create_small_test_config()

    # B10 — no print pollution during hook registration
    def test_no_stdout_during_hook_registration(self, capsys):
        from ironcore.layers.module import BaseModule

        config = self._make_config()

        class SampleModule(BaseModule):
            def forward(self, x):
                return x

        m = SampleModule(config)
        m.register_profile_hooks(gpu_profiler=True)
        captured = capsys.readouterr()
        assert captured.out == ""

    # B3 — unknown kwarg raises TypeError
    def test_unknown_kwarg_raises_type_error(self):
        from ironcore.layers.module import BaseModule

        config = self._make_config()

        class SampleModule(BaseModule):
            def forward(self, x):
                return x

        m = SampleModule(config)
        with pytest.raises(TypeError):
            m.register_profile_hooks(profile_nsys=True)  # type: ignore[call-arg]

    # B3 — correct kwargs do not raise
    def test_valid_kwargs_do_not_raise(self):
        from ironcore.layers.module import BaseModule

        config = self._make_config()

        class SampleModule(BaseModule):
            def forward(self, x):
                return x

        m = SampleModule(config)
        m.register_profile_hooks(gpu_profiler=True, torch_profiler=False, layer_timing=False)

    # B8 — hooks reach nested BaseModule children
    def test_hooks_reach_all_nested_base_module_children(self):
        from ironcore.layers.module import BaseModule

        config = self._make_config()

        class Leaf(BaseModule):
            def forward(self, x):
                return x

        class Middle(BaseModule):
            def __init__(self, cfg):
                super().__init__(cfg)
                self.leaf = Leaf(cfg)

            def forward(self, x):
                return self.leaf(x)

        class Root(BaseModule):
            def __init__(self, cfg):
                super().__init__(cfg)
                self.mid = Middle(cfg)

            def forward(self, x):
                return self.mid(x)

        root = Root(config)
        root.register_profile_hooks(layer_timing=True)

        assert root._hooks_registered
        assert root.mid._hooks_registered
        assert root.mid.leaf._hooks_registered

    # B8 — non-BaseModule children are not given profile hooks
    def test_hooks_skip_non_base_module_children(self):
        from torch import nn

        from ironcore.layers.module import BaseModule

        config = self._make_config()

        class WithPlainChild(BaseModule):
            def __init__(self, cfg):
                super().__init__(cfg)
                self.plain = nn.Linear(4, 4)  # not a BaseModule

            def forward(self, x):
                return self.plain(x)

        m = WithPlainChild(config)
        m.register_profile_hooks(layer_timing=True)
        # plain Linear has no _hooks_registered attribute
        assert not hasattr(m.plain, "_hooks_registered")

    # Idempotent — double registration must not double hooks
    def test_register_profile_hooks_is_idempotent(self):
        from ironcore.layers.module import BaseModule

        config = self._make_config()

        class SampleModule(BaseModule):
            def forward(self, x):
                return x

        m = SampleModule(config)
        m.register_profile_hooks(gpu_profiler=True)
        n_hooks_after_first = len(m._forward_pre_hooks)
        m.register_profile_hooks(gpu_profiler=True)
        assert len(m._forward_pre_hooks) == n_hooks_after_first

    # F6 / B3 — layer_timing sets _timing_collector
    def test_layer_timing_sets_timing_collector(self):
        _reset_singletons()
        from ironcore.layers.module import BaseModule

        config = self._make_config()

        class SampleModule(BaseModule):
            def forward(self, x):
                return x

        m = SampleModule(config)
        m.register_profile_hooks(layer_timing=True)
        assert m._timing_collector is not None
        assert m._timing_collector is get_layer_timing_collector()
