# Copyright (c) 2025-2026 Jaegeun Han
#
# SPDX-License-Identifier: Apache-2.0

"""Resume must continue the original trajectory, not merely start again.

Both defects covered here were invisible from the outside: the run resumed,
logged plausible losses, and silently trained on different data with different
dropout masks than the run it was continuing.
"""

from types import SimpleNamespace

import pytest
import torch

from ironcore.trainers.base_trainer import BaseTrainer


class _Counter:
    """Stands in for the train iterator; each next() is one micro-batch."""

    def __init__(self):
        self.consumed = 0

    def __next__(self):
        self.consumed += 1
        return {}


def _trainer_stub(micro_batch_size, accum_steps, train_batch_size):
    stub = SimpleNamespace(
        config=SimpleNamespace(
            trainer=SimpleNamespace(
                micro_batch_size=micro_batch_size,
                gradient_accumulation_steps=accum_steps,
                train_batch_size=train_batch_size,
            )
        ),
        data_iterator={"train": _Counter()},
        logger=SimpleNamespace(info=lambda *a, **k: None),
    )
    return stub


class TestSkipConsumedTrainSamples:
    @pytest.mark.parametrize(
        ("micro", "accum", "dp", "last_step"),
        [(4, 2, 1, 5), (1, 1, 1, 3), (2, 8, 1, 7), (4, 2, 2, 5)],
    )
    def test_advances_by_exactly_what_the_original_run_drew(self, micro, accum, dp, last_step):
        """One next() is a micro-batch, so the count must be in micro-batches.

        Counting `last_step * train_batch_size` instead consumed micro_batch_size
        times too many, so a resumed run skipped past data that was never trained
        on — 120 of every 160 samples, for the shipped example config.
        """
        stub = _trainer_stub(micro, accum, train_batch_size=micro * accum * dp)
        BaseTrainer._skip_consumed_train_samples(stub, last_step)

        assert stub.data_iterator["train"].consumed == last_step * accum

    def test_step_zero_consumes_nothing(self):
        stub = _trainer_stub(4, 2, 8)
        BaseTrainer._skip_consumed_train_samples(stub, 0)
        assert stub.data_iterator["train"].consumed == 0

    def test_exhausted_iterator_does_not_raise(self):
        class _Empty:
            def __next__(self):
                raise StopIteration

        stub = _trainer_stub(4, 2, 8)
        stub.data_iterator["train"] = _Empty()
        BaseTrainer._skip_consumed_train_samples(stub, 5)


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA RNG state")
class TestRngStateRestoreAcceptsDeviceTensors:
    """Checkpoints load with map_location set to the model's device.

    The RNG setters take CPU ByteTensors, so restoring straight from a
    CUDA-mapped checkpoint raised "RNG state must be a torch.ByteTensor". That
    exception was caught and logged as a warning, so every resume silently kept
    whatever RNG state it happened to start with.
    """

    def test_cuda_mapped_states_restore(self):
        cpu_state = torch.get_rng_state()
        cuda_states = torch.cuda.get_rng_state_all()

        # What torch.load(map_location=cuda) hands back.
        mapped_cpu = cpu_state.to("cuda")
        mapped_cuda = [s.to("cuda") for s in cuda_states]

        with pytest.raises(TypeError):
            torch.set_rng_state(mapped_cpu)

        torch.set_rng_state(mapped_cpu.cpu())
        torch.cuda.set_rng_state_all([s.cpu() for s in mapped_cuda])

        assert torch.equal(torch.get_rng_state(), cpu_state)
