# Copyright (c) 2025-2026 Jaegeun Han
#
# SPDX-License-Identifier: MIT
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the above copyright notice,
# this list of conditions, and the following disclaimer are retained.
#
# Full license text is available at LICENSE file.

"""Model-related fixtures for testing."""

from __future__ import annotations

import pytest
import torch
from tests.fixtures.config_fixtures import create_small_test_config, create_test_config

from ironcore.layers.attention import Attention
from ironcore.models.transformer import TransformerModel
from ironcore.parallel import parallel_states

# =============================================================================
# Device Fixtures
# =============================================================================


@pytest.fixture
def device() -> torch.device:
    """Fixture providing the best available device."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def cuda_device() -> torch.device | None:
    """Fixture providing CUDA device if available."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    return torch.device("cuda:0")


# =============================================================================
# Attention Fixtures
# =============================================================================


@pytest.fixture
def attention_layer(device: torch.device) -> Attention:
    """Fixture providing a basic attention layer."""
    config = create_test_config()
    attention = Attention(config).to(device)
    attention.init_weights()
    return attention


@pytest.fixture
def small_attention_layer(device: torch.device) -> Attention:
    """Fixture providing a small attention layer for fast testing."""
    config = create_small_test_config()
    attention = Attention(config).to(device)
    attention.init_weights()
    return attention


@pytest.fixture
def gqa_attention_layer(device: torch.device) -> Attention:
    """Fixture providing a GQA attention layer."""
    from tests.fixtures.config_fixtures import create_gqa_config

    config = create_gqa_config()
    attention = Attention(config).to(device)
    attention.init_weights()
    return attention


@pytest.fixture
def mqa_attention_layer(device: torch.device) -> Attention:
    """Fixture providing a MQA attention layer."""
    from tests.fixtures.config_fixtures import create_mqa_config

    config = create_mqa_config()
    attention = Attention(config).to(device)
    attention.init_weights()
    return attention


# =============================================================================
# Model Fixtures
# =============================================================================


@pytest.fixture
def transformer_model(device: torch.device) -> TransformerModel:
    """Fixture providing a transformer model."""
    config = create_small_test_config()

    # Initialize parallel states for testing
    try:
        parallel_states.get_tensor_model_parallel_world_size()
    except AssertionError:
        parallel_states.initialize_model_parallel(
            tensor_model_parallel_size=1,
            timeout_in_minutes=1.0,
        )

    model = TransformerModel(config).to(device)
    model.init_weights()
    return model


@pytest.fixture
def transformer_model_bf16(device: torch.device) -> TransformerModel:
    """Fixture providing a transformer model in bfloat16."""
    config = create_small_test_config(precision="bfloat16")

    try:
        parallel_states.get_tensor_model_parallel_world_size()
    except AssertionError:
        parallel_states.initialize_model_parallel(
            tensor_model_parallel_size=1,
            timeout_in_minutes=1.0,
        )

    model = TransformerModel(config).to(device=device, dtype=torch.bfloat16)
    model.init_weights()
    return model


# =============================================================================
# Input Fixtures
# =============================================================================


@pytest.fixture
def random_hidden_states(device: torch.device) -> torch.Tensor:
    """Fixture providing random hidden states."""
    config = create_small_test_config()
    return torch.randn(2, 32, config.model.d_model, device=device)


@pytest.fixture
def causal_attention_mask(device: torch.device) -> torch.Tensor:
    """Fixture providing a causal attention mask."""
    seq_len = 32
    return (
        torch.tril(torch.ones(seq_len, seq_len, device=device))
        .unsqueeze(0)
        .unsqueeze(0)
        .expand(2, -1, -1, -1)
    )


@pytest.fixture
def random_input_ids(device: torch.device) -> torch.Tensor:
    """Fixture providing random input IDs."""
    vocab_size = 1000
    return torch.randint(0, vocab_size, (2, 32), device=device)


# =============================================================================
# State Dict Fixtures
# =============================================================================


@pytest.fixture
def gpt2_state_dict() -> dict[str, torch.Tensor]:
    """Create a mock GPT-2 state dict for testing."""
    hidden_size = 64
    num_layers = 2
    vocab_size = 100

    state_dict = {
        "transformer.wte.weight": torch.randn(vocab_size, hidden_size),
        "transformer.wpe.weight": torch.randn(512, hidden_size),
        "transformer.ln_f.weight": torch.randn(hidden_size),
        "transformer.ln_f.bias": torch.randn(hidden_size),
    }

    for i in range(num_layers):
        prefix = f"transformer.h.{i}"
        state_dict[f"{prefix}.ln_1.weight"] = torch.randn(hidden_size)
        state_dict[f"{prefix}.ln_1.bias"] = torch.randn(hidden_size)
        state_dict[f"{prefix}.ln_2.weight"] = torch.randn(hidden_size)
        state_dict[f"{prefix}.ln_2.bias"] = torch.randn(hidden_size)
        state_dict[f"{prefix}.attn.c_attn.weight"] = torch.randn(hidden_size, 3 * hidden_size)
        state_dict[f"{prefix}.attn.c_attn.bias"] = torch.randn(3 * hidden_size)
        state_dict[f"{prefix}.attn.c_proj.weight"] = torch.randn(hidden_size, hidden_size)
        state_dict[f"{prefix}.attn.c_proj.bias"] = torch.randn(hidden_size)
        state_dict[f"{prefix}.mlp.c_fc.weight"] = torch.randn(hidden_size, 4 * hidden_size)
        state_dict[f"{prefix}.mlp.c_fc.bias"] = torch.randn(4 * hidden_size)
        state_dict[f"{prefix}.mlp.c_proj.weight"] = torch.randn(4 * hidden_size, hidden_size)
        state_dict[f"{prefix}.mlp.c_proj.bias"] = torch.randn(hidden_size)

    return state_dict


@pytest.fixture
def llama_state_dict() -> dict[str, torch.Tensor]:
    """Create a mock LLaMA state dict for testing."""
    hidden_size = 64
    num_layers = 2
    vocab_size = 100
    num_kv_heads = 2
    kv_dim = (hidden_size // 8) * num_kv_heads

    state_dict = {
        "model.embed_tokens.weight": torch.randn(vocab_size, hidden_size),
        "model.norm.weight": torch.randn(hidden_size),
        "lm_head.weight": torch.randn(vocab_size, hidden_size),
    }

    for i in range(num_layers):
        prefix = f"model.layers.{i}"
        state_dict[f"{prefix}.input_layernorm.weight"] = torch.randn(hidden_size)
        state_dict[f"{prefix}.post_attention_layernorm.weight"] = torch.randn(hidden_size)
        state_dict[f"{prefix}.self_attn.q_proj.weight"] = torch.randn(hidden_size, hidden_size)
        state_dict[f"{prefix}.self_attn.k_proj.weight"] = torch.randn(kv_dim, hidden_size)
        state_dict[f"{prefix}.self_attn.v_proj.weight"] = torch.randn(kv_dim, hidden_size)
        state_dict[f"{prefix}.self_attn.o_proj.weight"] = torch.randn(hidden_size, hidden_size)
        state_dict[f"{prefix}.mlp.gate_proj.weight"] = torch.randn(4 * hidden_size, hidden_size)
        state_dict[f"{prefix}.mlp.up_proj.weight"] = torch.randn(4 * hidden_size, hidden_size)
        state_dict[f"{prefix}.mlp.down_proj.weight"] = torch.randn(hidden_size, 4 * hidden_size)

    return state_dict


# =============================================================================
# FIM (Fill-In-the-Middle) Fixtures
# =============================================================================


@pytest.fixture
def test_tokenizer_with_fim():
    """HuggingFace tokenizer with FIM special tokens added."""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.add_special_tokens(
        {"additional_special_tokens": ["<fim_prefix>", "<fim_suffix>", "<fim_middle>"]}
    )
    return tokenizer


@pytest.fixture
def test_tokenizer_without_fim():
    """HuggingFace tokenizer without FIM tokens (for error testing)."""
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained("gpt2")


@pytest.fixture
def fim_token_ids(test_tokenizer_with_fim):
    """FIM special token IDs."""
    return {
        "prefix": test_tokenizer_with_fim.convert_tokens_to_ids("<fim_prefix>"),
        "suffix": test_tokenizer_with_fim.convert_tokens_to_ids("<fim_suffix>"),
        "middle": test_tokenizer_with_fim.convert_tokens_to_ids("<fim_middle>"),
    }
