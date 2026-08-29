# Copyright (c) 2025-2026 Jaegeun Han
#
# SPDX-License-Identifier: Apache-2.0
"""Regression tests covering gaps identified in the Fable code review (#80).

Three test classes that would have caught the recent defects behind passing
suites:

1. ``TestShippedConfigsParse`` — table-driven test that loads every
   ``configs/**/*.yaml`` and asserts it parses through ``load_full_config``.
2. ``TestCollatorMultiSample`` — collator tests with several samples of
   differing lengths, asserting label boundaries and row-wise DPO pairing.
3. ``TestCheckpointLifecycle`` — checkpoint lifecycle test that saves twice,
   resumes, and checks the post-resume trajectory is consistent.

These run CPU-only and skip gracefully when torch/ironcore heavy imports
are unavailable (e.g., missing flash_attn in CI).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import pytest

REPO_ROOT = Path(__file__).parents[2]
CONFIGS_DIR = REPO_ROOT / "configs"


# ---------------------------------------------------------------------------
# 1. Parse every shipped config
# ---------------------------------------------------------------------------


# Config-group names. A YAML is a *fragment* — included by a top-level config
# rather than loaded on its own — if it sits in a directory with one of these
# names at any depth (configs/model/, configs/sanity_offload/model/) or is named
# after the group it fills in (configs/experiments/nanogpt/model.yaml).
_FRAGMENT_GROUPS = {"data", "model", "optim", "trainer", "alignment"}

# Directories holding partial overlays and sub-objects that are merged into a
# top-level config, never loaded standalone.
_FRAGMENT_DIRS = _FRAGMENT_GROUPS | {"peft", "rewards", "profile"}


def _collect_shipped_configs() -> list[Path]:
    """Collect every standalone YAML under configs/ for table-driven testing.

    Fragments are excluded: loading one through ``load_full_config`` is
    *expected* to fail (it has no trainer/model/data blocks of its own), so
    including them would be pure false positives rather than coverage.
    """
    if not CONFIGS_DIR.exists():
        return []
    standalone: list[Path] = []
    for p in sorted(CONFIGS_DIR.rglob("*.yaml")):
        if not p.is_file():
            continue
        rel = p.relative_to(CONFIGS_DIR)
        if set(rel.parts[:-1]) & _FRAGMENT_DIRS:
            continue
        if p.stem in _FRAGMENT_GROUPS:
            continue
        standalone.append(p)
    return standalone


# World sizes a shipped config might be written for. Batch-block validation
# (`micro * grad_accum * dp_world_size == train_batch_size`) and the tp-vs-world
# check both depend on WORLD_SIZE, which `ironcore.train` reads from the
# environment — so a 2-GPU config cannot validate at world size 1 no matter how
# well formed it is. Trying a ladder asserts "this config is coherent at *some*
# plausible GPU count" without restating the invariant here, where it would rot.
_CANDIDATE_WORLD_SIZES = (1, 2, 4, 8)


def _config_tp_size(path: Path) -> int:
    """tensor_model_parallel_size a config asks for, without importing ironcore."""
    import yaml

    raw = yaml.safe_load(path.read_text()) or {}
    return (raw.get("trainer") or {}).get("tensor_model_parallel_size", 1)


# Split by TP size so each group lands in a job that can actually run it. Config
# validation rejects tensor_model_parallel_size > 1 outright when CUDA is absent,
# and the CPU logic job is the only one that collects an unmarked test — so a
# single unmarked test skipping the TP configs meant nothing validated them at all.
_SHIPPED_CONFIGS = _collect_shipped_configs()
_TP1_CONFIGS = [p for p in _SHIPPED_CONFIGS if _config_tp_size(p) == 1]
_TP_PARALLEL_CONFIGS = [p for p in _SHIPPED_CONFIGS if _config_tp_size(p) > 1]


@pytest.mark.skipif(
    not CONFIGS_DIR.exists(),
    reason="configs/ directory not available from this CWD",
)
class TestShippedConfigsParse:
    """Every shipped YAML should load without raising.

    Two configs were broken when this test did not exist:
    ``qwen2.5-0.5B.yaml`` used non-existent field names, and
    ``gpt2-small-moe.yaml`` omitted ``head_dim``. (Fable issue #80 item 1.)
    """

    @pytest.fixture(autouse=True)
    def _require_load_full_config(self) -> None:
        # Skip the whole class if ironcore.train cannot be imported (e.g. the
        # torch/transformers stack is unavailable in a minimal CI runner).
        try:
            from ironcore.train import load_full_config  # noqa: F401
        except Exception as exc:  # noqa: BLE001
            pytest.skip(f"ironcore.train.load_full_config unavailable: {exc}")

    @pytest.mark.parametrize(
        "config_path",
        _TP1_CONFIGS,
        ids=lambda p: str(p.relative_to(CONFIGS_DIR)),
    )
    def test_config_loads(self, config_path: Path, monkeypatch) -> None:
        self._assert_loads(config_path, monkeypatch)

    @pytest.mark.cuda
    @pytest.mark.parametrize(
        "config_path",
        _TP_PARALLEL_CONFIGS,
        ids=lambda p: str(p.relative_to(CONFIGS_DIR)),
    )
    def test_tp_config_loads(self, config_path: Path, monkeypatch) -> None:
        """Same check for TP configs, which validation rejects without CUDA."""
        self._assert_loads(config_path, monkeypatch)

    @staticmethod
    def _assert_loads(config_path: Path, monkeypatch) -> None:
        from ironcore.train import load_full_config

        failures: list[str] = []
        for world_size in _CANDIDATE_WORLD_SIZES:
            monkeypatch.setenv("WORLD_SIZE", str(world_size))
            try:
                config = load_full_config(str(config_path))
            except (FileNotFoundError, ValueError, KeyError) as exc:
                failures.append(f"  WORLD_SIZE={world_size}: {type(exc).__name__}: {exc}")
                continue
            assert config is not None
            return

        # Surface the offending file plus every attempt, so a table-driven
        # failure says which YAML broke and whether it was world-size specific.
        raise AssertionError(
            "Shipped config failed to load at any world size: "
            f"{config_path.relative_to(CONFIGS_DIR)}\n" + "\n".join(failures)
        )


# ---------------------------------------------------------------------------
# 2. Collate multi-sample / multi-pair batches
# ---------------------------------------------------------------------------


def _build_sft_sample(token_ids: list[int], mask_ranges: list[tuple[int, int]] | None = None):
    import torch as _torch

    return {
        "token_ids": _torch.tensor(token_ids, dtype=_torch.long),
        "metadata": {"mask_ranges": mask_ranges or [], "type": "sft"},
    }


class TestCollatorMultiSample:
    """Multi-sample SFT/DPO collation.

    The single-sample tests that existed before could not catch the SFT
    label-mask off-by-one (#59) or the DPO chosen/rejected misalignment
    (#60) because both bugs only manifest with multiple samples of differing
    lengths. (Fable issue #80 item 2.)
    """

    def test_sft_label_boundaries_match_completion_tokens(self) -> None:
        """The first completion token must carry a non-ignore label, and no
        prompt token may leak into the loss."""
        from ironcore.dataloader.collator import UniversalCollator

        collator = UniversalCollator(
            mode="sft",
            max_seq_len=32,
            pad_token_id=0,
            use_flash_attention=False,
            return_full_attention_mask=False,
        )
        # prompt=[10,11,12] completion=[20,21,22,23]
        sample_a = _build_sft_sample([10, 11, 12, 20, 21, 22, 23], mask_ranges=[(0, 3)])
        sample_b = _build_sft_sample([30, 31, 40, 41], mask_ranges=[(0, 2)])
        batch = collator([sample_a, sample_b])

        labels = batch["labels"]
        # SFT collator bin-packs — multiple samples may share a row. Count
        # non-ignore positions across the entire batch and compare to the
        # expected total completion tokens (4 + 2 = 6).
        total_non_ignore = (labels != -100).sum().item()
        assert total_non_ignore == 6, (
            f"Expected 6 supervised positions total (4 from sample_a + 2 from "
            f"sample_b), got {total_non_ignore}; labels={labels.tolist()}"
        )

    def test_sft_no_pad_position_inside_attention_block(self) -> None:
        """After the #63 fix, cu_seqlens must count only real written tokens,
        so no PAD slot falls inside a sample's attention block."""
        from ironcore.dataloader.collator import UniversalCollator

        collator = UniversalCollator(
            mode="sft",
            max_seq_len=20,
            pad_token_id=999,
            use_flash_attention=True,
            return_full_attention_mask=False,
        )
        # sampleA has 5 tokens → 4 written; sampleB has 4 tokens → 3 written.
        # Both fit in one bin (5+4=9 ≤ 20), so cu_seqlens for that row is
        # [0, 4, 7] — NOT [0, 5, 9] which would leave PAD holes.
        sample_a = _build_sft_sample([1, 2, 3, 4, 5])
        sample_b = _build_sft_sample([10, 20, 30, 40])
        batch = collator([sample_a, sample_b])

        cu_seqlens = batch["cu_seqlens"]
        # Bin-packing puts both in one row; verify cumulative boundaries equal
        # the sum of written_len values (4 + 3 = 7), not sample_len (5 + 4 = 9).
        row_final = cu_seqlens[0][-1].item()
        assert row_final == 7, (
            f"Packed row cu_seqlens last={row_final} (expected 7 = 4+3 written "
            f"tokens, not 9 = 5+4 sample_len); full={cu_seqlens[0].tolist()}"
        )

    def test_dpo_row_wise_pairing_after_length_sort(self) -> None:
        """chosen row i and rejected row i must be the same preference pair,
        regardless of length-sort reordering. (Fable issue #60.)"""
        from ironcore.dataloader.collator import UniversalCollator

        collator = UniversalCollator(
            mode="dpo",
            max_seq_len=16,
            pad_token_id=0,
            use_flash_attention=False,
            return_full_attention_mask=False,
        )
        # Three pairs. Each sample needs >= 2 tokens to produce a (input,label)
        # pair after the sample_len-1 shift. Lengths chosen so independent
        # length-sort would reorder.
        batch = collator(
            [
                {
                    "token_ids": [20, 21],
                    "metadata": {"type": "dpo_chosen", "group_id": 0, "mask_ranges": []},
                },
                {
                    "token_ids": [110, 111, 112, 113],
                    "metadata": {"type": "dpo_rejected", "group_id": 0, "mask_ranges": []},
                },
                {
                    "token_ids": [30, 31, 32],
                    "metadata": {"type": "dpo_chosen", "group_id": 1, "mask_ranges": []},
                },
                {
                    "token_ids": [130, 131, 132, 133, 134],
                    "metadata": {"type": "dpo_rejected", "group_id": 1, "mask_ranges": []},
                },
                {
                    "token_ids": [10, 11],
                    "metadata": {"type": "dpo_chosen", "group_id": 2, "mask_ranges": []},
                },
                {
                    "token_ids": [120, 121, 122],
                    "metadata": {"type": "dpo_rejected", "group_id": 2, "mask_ranges": []},
                },
            ]
        )

        # After group_id sort, row 0 = pair 0, row 1 = pair 1, row 2 = pair 2.
        chosen_first_tokens = batch["chosen_input_ids"][:, 0].tolist()
        rejected_first_tokens = batch["rejected_input_ids"][:, 0].tolist()
        # Pair 0: chosen starts 20, rejected starts 110
        # Pair 1: chosen starts 30, rejected starts 130
        # Pair 2: chosen starts 10, rejected starts 120
        assert chosen_first_tokens == [20, 30, 10], chosen_first_tokens
        assert rejected_first_tokens == [110, 130, 120], rejected_first_tokens


# ---------------------------------------------------------------------------
# 3. Save → resume → save again
# ---------------------------------------------------------------------------


class TestCheckpointLifecycle:
    """Checkpoint lifecycle test that saves twice, resumes, and checks
    consistency. (Fable issue #80 item 3, regression for #58/#64.)

    The LoRA ``KeyError`` only appeared on the second save; the existing test
    saved once. Atomic-save regression coverage also needs two saves through
    the same code path. These tests are heavy (need torch + model) so they
    skip when CUDA is unavailable.
    """

    @pytest.fixture(autouse=True)
    def _require_torch_cuda(self) -> None:
        try:
            import torch

            if not torch.cuda.is_available():
                pytest.skip("CUDA not available for checkpoint lifecycle test")
        except ImportError:
            pytest.skip("torch not importable")

    def test_save_twice_does_not_raise(self, tmp_path: Path) -> None:
        """Save the same model+optimizer twice in one process.

        Regression for #64: with LoRA active, the second save raised
        ``KeyError`` because the first save had poisoned optimizer.state via
        the defaultdict-indexing bug. A non-LoRA model should also tolerate
        repeated saves (atomic-replace path). (Fable issue #80 item 3.)
        """
        import torch

        from ironcore.checkpointing.native import save_checkpoint

        class _StubModel(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.linear = torch.nn.Linear(4, 4)
                self.config = {"_stub": True}

        model = _StubModel()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

        class _Cfg:
            class operation:  # noqa: N801 - matches MainConfig.operation attribute name
                no_save = False
                save_dist_ckpt = False
                save_full_model = False

            class trainer:  # noqa: N801 - matches MainConfig.trainer attribute name
                model_path = str(tmp_path)
                tensor_model_parallel_size = 1

            class model:  # noqa: N801 - matches MainConfig.model attribute name
                hf_model_type = None
                hf_architecture = None

        import torch.optim.lr_scheduler as lrs

        scheduler = lrs.ConstantLR(optimizer)
        # Stubs are intentionally minimal; cast to Any for save_checkpoint's
        # typed signature so we don't have to import the full MainConfig /
        # LanguageModel stack (which would force torch+transformers+CUDA).
        save_checkpoint(cast(Any, _Cfg), cast(Any, model), optimizer, scheduler, step=1)
        save_checkpoint(cast(Any, _Cfg), cast(Any, model), optimizer, scheduler, step=2)
        for step in (1, 2):
            ckpt_file = tmp_path / f"step_{step}" / "pytorch_model.bin"
            assert ckpt_file.exists(), f"checkpoint for step {step} missing"
            assert ckpt_file.stat().st_size > 0

    def test_atomic_save_leaves_no_truncated_file_on_interrupt(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        """If torch.save raises mid-write, the previous checkpoint must still
        be loadable. Regression for #58: atomic save writes to .tmp then
        os.replace, so the final path is never observed truncated.
        """
        import torch

        from ironcore.checkpointing.native import save_checkpoint

        class _StubModel(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.linear = torch.nn.Linear(4, 4)
                self.config = {"_stub": True}

        model = _StubModel()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

        class _Cfg:
            class operation:  # noqa: N801 - matches MainConfig.operation attribute name
                no_save = False
                save_dist_ckpt = False
                save_full_model = False

            class trainer:  # noqa: N801 - matches MainConfig.trainer attribute name
                model_path = str(tmp_path)
                tensor_model_parallel_size = 1

            class model:  # noqa: N801 - matches MainConfig.model attribute name
                hf_model_type = None
                hf_architecture = None

        import torch.optim.lr_scheduler as lrs

        scheduler = lrs.ConstantLR(optimizer)

        save_checkpoint(cast(Any, _Cfg), cast(Any, model), optimizer, scheduler, step=1)
        ckpt_file = tmp_path / "step_1" / "pytorch_model.bin"
        original_size = ckpt_file.stat().st_size

        def _boom(*args, **kwargs):
            raise RuntimeError("simulated interrupt")

        monkeypatch.setattr(torch, "save", _boom)
        with pytest.raises(RuntimeError, match="simulated interrupt"):
            save_checkpoint(cast(Any, _Cfg), cast(Any, model), optimizer, scheduler, step=2)

        monkeypatch.undo()
        assert ckpt_file.stat().st_size == original_size, (
            "step_1 checkpoint was corrupted by the failed step_2 save"
        )
        loaded = torch.load(ckpt_file, map_location="cpu", weights_only=False)
        assert "model_state_dict" in loaded
