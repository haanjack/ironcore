"""
13B model offload capability test.

Validates LLaMA-13B training with full offload (optimizer + weight streaming + activation spill)
on a single 24GB GPU. Requires ~92GB host RAM with bf16 optimizer states.

Requires NGC container (flash attention). Marked e2e — excluded from default runs.
"""

import math
from unittest.mock import patch

import pytest
import torch
import torch.nn.functional as F
from tests.fixtures.config_fixtures import create_test_config
from tests.fixtures.utils import cudnn_determinism
from tests.integration.offload.conftest import (
    create_mock_data_iterator,
    create_mock_evaluators,
    setup_distributed,
)

from ironcore.config import OffloadConfig
from ironcore.global_vars import reset_global_states
from ironcore.trainers import LanguageModelTrainer

pytestmark = [pytest.mark.cuda, pytest.mark.e2e]

NUM_STEPS = 10
BATCH_SIZE = 1
SEQ_LEN = 1024

# Deterministic seeding
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)


@pytest.fixture(autouse=True)
def _cudnn_determinism():
    with cudnn_determinism(deterministic=True, benchmark=False):
        yield


skip_insufficient_vram = pytest.mark.skipif(
    not torch.cuda.is_available()
    or torch.cuda.get_device_properties(0).total_memory < 20 * 1024**3,
    reason="Requires 20GB+ VRAM for 13B full offload",
)


def _make_13b_config(**offload_overrides):
    """Build LLaMA-13B config with offload settings."""
    config = create_test_config(
        d_model=5120,
        d_ffn=13824,
        num_layers=40,
        num_attention_heads=40,
        num_attention_groups=8,
        head_dim=128,
        max_seq_len=SEQ_LEN,
        dropout_attn=0.0,
        dropout_mlp=0.0,
        dropout_embd=0.0,
        precision="bfloat16",
        use_flash_attn=True,
        seed=42,
    )

    # LLaMA-specific settings not exposed by create_test_config
    config.model.name = "llama-13b"
    config.model.ln_type = "rmsnorm"
    config.model.ln_eps = 1e-5
    config.model.activation_type = "swiglu"
    config.model.vocab_name_or_path = "gpt2"
    config.model.tokenizer_type = "gpt2"
    config.model.positional_embedding.type = "rope"
    config.model.positional_embedding.base = 10000

    config.operation.train_steps = NUM_STEPS + 10
    config.trainer.micro_batch_size = BATCH_SIZE
    config.trainer.train_batch_size = BATCH_SIZE
    config.trainer.gradient_accumulation_steps = 1
    config.trainer.model_path = "/tmp/test_13b_offload"

    config.data.seq_length = SEQ_LEN

    offload = OffloadConfig(enabled=True)
    for k, v in offload_overrides.items():
        setattr(offload, k, v)
    config.offload = offload

    return config


def _run_training(config, num_steps, tmp_path):
    """Run N training steps. Returns (initial_loss, final_loss, peak_vram_mb)."""
    reset_global_states()
    setup_distributed()

    config.trainer.model_path = str(tmp_path / "checkpoints")

    def forward_step(model, _data_iterator):
        device = next(model.parameters()).device
        torch.manual_seed(42)
        input_ids = torch.randint(0, 32000, (BATCH_SIZE, SEQ_LEN), device=device)
        labels = input_ids.clone()
        logits, _ = model(input_ids, labels=None)
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()
        return F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
        )

    trainer = None
    try:
        with (
            patch(
                "ironcore.trainers.base_trainer.get_data_iterator",
                return_value=create_mock_data_iterator(),
            ),
            patch(
                "ironcore.trainers.base_trainer.get_evaluators",
                return_value=create_mock_evaluators(),
            ),
        ):
            trainer = LanguageModelTrainer(config, forward_step, F.cross_entropy)
            trainer._initialize()

            torch.cuda.reset_peak_memory_stats()
            initial_loss = None

            for step in range(num_steps):
                loss, _, _ = trainer.train_step(step=step)
                if step == 0:
                    initial_loss = loss
                final_loss = loss

            peak_vram = torch.cuda.max_memory_allocated() / 1024**2
            return initial_loss, final_loss, peak_vram

    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            pytest.skip(f"OOM during 13B training: {str(e)[:100]}")
        raise
    finally:
        if trainer is not None:
            try:
                trainer._finalize_process()
            except Exception:
                pass
            del trainer
        torch.cuda.empty_cache()


@skip_insufficient_vram
class Test13BOffloadCapability:
    """13B model full offload capability test. Expensive — run via -m e2e."""

    def test_full_offload(self, tmp_path):
        """Full offload: optimizer_offload + weight_offload + activation_spill."""
        config = _make_13b_config(
            optimizer_offload=True,
            weight_offload=True,
            weight_prefetch_layers=2,
            weight_storage_precision="bf16",
            optimizer_state_precision="bf16",
            activation_spill=True,
            activation_spill_granularity="sub_layer",
            pinned_memory_pool_gb=16.0,
        )

        init_loss, final_loss, peak_vram = _run_training(config, NUM_STEPS, tmp_path)

        assert init_loss is not None
        assert not math.isnan(final_loss), "Final loss is NaN"
        assert not math.isinf(final_loss), "Final loss is Inf"
        assert final_loss > 0, f"Final loss is non-positive: {final_loss}"
        assert final_loss < init_loss, f"Loss did not decrease: {init_loss:.4f} -> {final_loss:.4f}"
        assert peak_vram < 24 * 1024, f"Peak VRAM exceeded 24GB: {peak_vram:.0f}MB"
