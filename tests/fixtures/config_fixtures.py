# Copyright (c) 2025-2026 Jaegeun Han
#
# SPDX-License-Identifier: Apache-2.0

"""Configuration fixtures and presets for testing."""

from __future__ import annotations

import pytest

from ironcore.config import (
    DataConfig,
    InitConfig,
    MainConfig,
    ModelConfig,
    OperationConfig,
    OptimConfig,
    ParallelConfig,
    PEFTConfig,
    ProfilerConfig,
    TrainerConfig,
    UtilsConfig,
)
from ironcore.config.config_alignment import AlignmentConfig

# =============================================================================
# Configuration Presets
# =============================================================================

# Small model config for fast testing
SMALL_MODEL_CONFIG = {
    "d_model": 128,
    "num_attention_heads": 4,
    "num_attention_groups": 4,
    "head_dim": 32,
    "d_ffn": 256,
    "num_layers": 2,
    "max_seq_len": 64,
    "dropout_attn": 0.0,
    "dropout_mlp": 0.0,
    "dropout_embd": 0.0,
    "no_bias": False,
    "precision": "float32",
}

# Standard model config for regular testing
STANDARD_MODEL_CONFIG = {
    "d_model": 512,
    "num_attention_heads": 8,
    "num_attention_groups": 8,
    "head_dim": 64,
    "d_ffn": 2048,
    "num_layers": 4,
    "max_seq_len": 128,
    "dropout_attn": 0.0,
    "dropout_mlp": 0.0,
    "dropout_embd": 0.0,
    "no_bias": False,
    "precision": "float32",
}


# =============================================================================
# Configuration Factories
# =============================================================================


def create_test_config(
    d_model: int = 512,
    num_attention_heads: int = 8,
    num_attention_groups: int = 8,
    head_dim: int = 64,
    d_ffn: int = 2048,
    num_layers: int = 4,
    max_seq_len: int = 128,
    max_position_embeddings: int | None = None,
    dropout_attn: float = 0.0,
    dropout_mlp: float = 0.0,
    dropout_embd: float = 0.0,
    no_bias: bool = False,
    precision: str = "float32",
    use_flash_attn: bool = False,
    tensor_model_parallel_size: int = 1,
    sequence_chunk_size: int | None = None,
    seed: int = 42,
    init_std: float = 0.02,
) -> MainConfig:
    """Create a test configuration with sensible defaults."""
    model_config = ModelConfig(
        d_model=d_model,
        num_attention_heads=num_attention_heads,
        num_attention_groups=num_attention_groups,
        head_dim=head_dim,
        d_ffn=d_ffn,
        num_layers=num_layers,
        max_seq_len=max_seq_len,
        max_position_embeddings=max_position_embeddings or max_seq_len,
        dropout_attn=dropout_attn,
        dropout_mlp=dropout_mlp,
        dropout_embd=dropout_embd,
        no_bias=no_bias,
        precision=precision,
    )

    trainer_config = TrainerConfig(
        tensor_model_parallel_size=tensor_model_parallel_size,
        use_flash_attn=use_flash_attn,
        sequence_chunk_size=sequence_chunk_size,
    )

    init_config = InitConfig(seed=seed, init_std=init_std)
    optim_config = OptimConfig()
    data_config = DataConfig()
    parallel_config = ParallelConfig()
    operation_config = OperationConfig()
    utils_config = UtilsConfig()
    peft_config = PEFTConfig()

    return MainConfig(
        model=model_config,
        trainer=trainer_config,
        init=init_config,
        optim=optim_config,
        data=data_config,
        parallel=parallel_config,
        operation=operation_config,
        utils=utils_config,
        peft=peft_config,
        profiler=ProfilerConfig(),
        alignment=AlignmentConfig(),
    )


def create_small_test_config(**kwargs) -> MainConfig:
    """Create a small test configuration for fast testing."""
    return create_test_config(**{**SMALL_MODEL_CONFIG, **kwargs})


def create_standard_test_config(**kwargs) -> MainConfig:
    """Create a standard test configuration."""
    return create_test_config(**{**STANDARD_MODEL_CONFIG, **kwargs})


def create_gqa_config(num_heads: int = 8, num_groups: int = 2, **kwargs) -> MainConfig:
    """Create a GQA (Grouped Query Attention) configuration."""
    return create_test_config(
        num_attention_heads=num_heads,
        num_attention_groups=num_groups,
        **kwargs,
    )


def create_mqa_config(num_heads: int = 8, **kwargs) -> MainConfig:
    """Create a MQA (Multi-Query Attention) configuration."""
    return create_test_config(
        num_attention_heads=num_heads,
        num_attention_groups=1,
        **kwargs,
    )


def create_tp_config(tp_size: int = 2, **kwargs) -> MainConfig:
    """Create a tensor parallel configuration."""
    return create_test_config(
        tensor_model_parallel_size=tp_size,
        **kwargs,
    )


# =============================================================================
# Pytest Fixtures
# =============================================================================


@pytest.fixture
def small_config() -> MainConfig:
    """Fixture providing a small test configuration."""
    return create_small_test_config()


@pytest.fixture
def standard_config() -> MainConfig:
    """Fixture providing a standard test configuration."""
    return create_standard_test_config()


@pytest.fixture
def gqa_config() -> MainConfig:
    """Fixture providing a GQA configuration."""
    return create_gqa_config()


@pytest.fixture
def mqa_config() -> MainConfig:
    """Fixture providing a MQA configuration."""
    return create_mqa_config()


@pytest.fixture(params=[8, 4, 1])
def attention_config(request) -> MainConfig:
    """Parametrized fixture for different attention group configurations."""
    num_groups = request.param
    return create_test_config(
        num_attention_heads=8,
        num_attention_groups=num_groups,
    )


# =============================================================================
# FIM (Fill-In-the-Middle) Configuration Fixtures
# =============================================================================


@pytest.fixture
def temp_dir():
    """Temporary directory for test outputs."""
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def test_config_fim_enabled(temp_dir):
    """Test configuration with FIM enabled at 50%."""
    from ironcore.dataloader.data_config import DatasetConfig, UniversalDataConfig

    return UniversalDataConfig(
        datasets=[
            DatasetConfig(
                name="test_fim",
                source="dummy",
                task_type="pretrain",
                text_column="text",
            )
        ],
        vocab_name_or_path="gpt2",
        seq_length=1024,
        preprocessed_dir=temp_dir / "preprocessed",
        cache_dir=temp_dir / "cache",
        # FIM settings at config level (not per-dataset)
        fim_rate=0.5,
        fim_prefix_token="<fim_prefix>",
        fim_suffix_token="<fim_suffix>",
        fim_middle_token="<fim_middle>",
    )


@pytest.fixture
def test_config_fim_disabled(temp_dir):
    """Test configuration with FIM disabled."""
    from ironcore.dataloader.data_config import DatasetConfig, UniversalDataConfig

    return UniversalDataConfig(
        datasets=[
            DatasetConfig(
                name="test_no_fim",
                source="dummy",
                task_type="pretrain",
                text_column="text",
            )
        ],
        vocab_name_or_path="gpt2",
        seq_length=1024,
        preprocessed_dir=temp_dir / "preprocessed",
        cache_dir=temp_dir / "cache",
        fim_rate=0.0,  # FIM disabled
    )


@pytest.fixture
def test_config_fim_100(temp_dir):
    """Test configuration with FIM at 100%."""
    from ironcore.dataloader.data_config import DatasetConfig, UniversalDataConfig

    return UniversalDataConfig(
        datasets=[
            DatasetConfig(
                name="test_fim_100",
                source="dummy",
                task_type="pretrain",
                text_column="text",
            )
        ],
        vocab_name_or_path="gpt2",
        seq_length=1024,
        preprocessed_dir=temp_dir / "preprocessed",
        cache_dir=temp_dir / "cache",
        # FIM at 100%
        fim_rate=1.0,
        fim_prefix_token="<fim_prefix>",
        fim_suffix_token="<fim_suffix>",
        fim_middle_token="<fim_middle>",
    )


@pytest.fixture
def serializer_with_fim(test_config_fim_enabled, test_tokenizer_with_fim):
    """DataSerializer with FIM-enabled tokenizer."""
    from ironcore.preprocessing.serializer import DataSerializer

    return DataSerializer(test_config_fim_enabled, test_tokenizer_with_fim, verbose=False)


@pytest.fixture
def serializer_without_fim(test_config_fim_disabled, test_tokenizer_without_fim):
    """DataSerializer with FIM disabled."""
    from ironcore.preprocessing.serializer import DataSerializer

    return DataSerializer(test_config_fim_disabled, test_tokenizer_without_fim, verbose=False)


@pytest.fixture
def serializer_fim_100(test_config_fim_100, test_tokenizer_with_fim):
    """DataSerializer with FIM at 100%."""
    from ironcore.preprocessing.serializer import DataSerializer

    return DataSerializer(test_config_fim_100, test_tokenizer_with_fim, verbose=False)
