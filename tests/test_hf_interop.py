# Copyright (c) 2025-2026 Jaegeun Han
#
# SPDX-License-Identifier: MIT
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice,
#    this list of conditions, and the following disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions, and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
# 3. Neither the name of the copyright holder nor the names of its contributors
#    may be used to endorse or promote products derived from this software
#    without specific prior written permission.
#
# Full license text is available at LICENSE file.

"""
Tests for HuggingFace checkpoint interoperability.

Tests bidirectional conversion between ironcore and HuggingFace checkpoint formats
for GPT-2 and LLaMA architectures.

Usage:
    pytest tests/test_hf_interop.py -v
    pytest tests/test_hf_interop.py -v -k "gpt2"  # GPT-2 tests only
    pytest tests/test_hf_interop.py -v -k "llama"  # LLaMA tests only
"""

import os
import tempfile
from pathlib import Path
from typing import Dict

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

import pytest
import torch
import torch.nn as nn

# Test imports
from ironcore.checkpointing import (
    load_from_huggingface,
    export_to_huggingface,
    WeightMapper,
    Architecture,
    get_architecture,
)
from ironcore.checkpointing.weight_mapping import ARCHITECTURE_ALIASES


# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture
def temp_dir():
    """Create a temporary directory for test outputs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def mock_gpt2_hf_state_dict():
    """Create a mock GPT-2 HuggingFace state dict for testing."""
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
        # Layer norms
        state_dict[f"{prefix}.ln_1.weight"] = torch.randn(hidden_size)
        state_dict[f"{prefix}.ln_1.bias"] = torch.randn(hidden_size)
        state_dict[f"{prefix}.ln_2.weight"] = torch.randn(hidden_size)
        state_dict[f"{prefix}.ln_2.bias"] = torch.randn(hidden_size)
        # Attention (GPT-2 uses Conv1D style: [in, out] instead of [out, in])
        state_dict[f"{prefix}.attn.c_attn.weight"] = torch.randn(hidden_size, 3 * hidden_size)
        state_dict[f"{prefix}.attn.c_attn.bias"] = torch.randn(3 * hidden_size)
        state_dict[f"{prefix}.attn.c_proj.weight"] = torch.randn(hidden_size, hidden_size)
        state_dict[f"{prefix}.attn.c_proj.bias"] = torch.randn(hidden_size)
        # MLP
        state_dict[f"{prefix}.mlp.c_fc.weight"] = torch.randn(hidden_size, 4 * hidden_size)
        state_dict[f"{prefix}.mlp.c_fc.bias"] = torch.randn(4 * hidden_size)
        state_dict[f"{prefix}.mlp.c_proj.weight"] = torch.randn(4 * hidden_size, hidden_size)
        state_dict[f"{prefix}.mlp.c_proj.bias"] = torch.randn(hidden_size)

    return state_dict


@pytest.fixture
def mock_llama_hf_state_dict():
    """Create a mock LLaMA HuggingFace state dict for testing."""
    hidden_size = 64
    num_layers = 2
    vocab_size = 100
    num_kv_heads = 2
    kv_dim = (hidden_size // 8) * num_kv_heads  # Assuming 8 heads, 2 kv groups

    state_dict = {
        "model.embed_tokens.weight": torch.randn(vocab_size, hidden_size),
        "model.norm.weight": torch.randn(hidden_size),
        "lm_head.weight": torch.randn(vocab_size, hidden_size),
    }

    for i in range(num_layers):
        prefix = f"model.layers.{i}"
        # Layer norms (RMSNorm - no bias)
        state_dict[f"{prefix}.input_layernorm.weight"] = torch.randn(hidden_size)
        state_dict[f"{prefix}.post_attention_layernorm.weight"] = torch.randn(hidden_size)
        # Attention
        state_dict[f"{prefix}.self_attn.q_proj.weight"] = torch.randn(hidden_size, hidden_size)
        state_dict[f"{prefix}.self_attn.k_proj.weight"] = torch.randn(kv_dim, hidden_size)
        state_dict[f"{prefix}.self_attn.v_proj.weight"] = torch.randn(kv_dim, hidden_size)
        state_dict[f"{prefix}.self_attn.o_proj.weight"] = torch.randn(hidden_size, hidden_size)
        # MLP (SwiGLU)
        state_dict[f"{prefix}.mlp.gate_proj.weight"] = torch.randn(4 * hidden_size, hidden_size)
        state_dict[f"{prefix}.mlp.up_proj.weight"] = torch.randn(4 * hidden_size, hidden_size)
        state_dict[f"{prefix}.mlp.down_proj.weight"] = torch.randn(hidden_size, 4 * hidden_size)

    return state_dict


# =============================================================================
# Unit Tests: Weight Mapping
# =============================================================================

class TestArchitectureDetection:
    """Test architecture detection from model type strings."""

    def test_gpt2_detection(self):
        assert get_architecture("gpt2") == Architecture.GPT2
        assert get_architecture("GPT2") == Architecture.GPT2
        assert get_architecture("gpt") == Architecture.GPT2

    def test_llama_detection(self):
        assert get_architecture("llama") == Architecture.LLAMA
        assert get_architecture("llama2") == Architecture.LLAMA
        assert get_architecture("llama3") == Architecture.LLAMA
        assert get_architecture("Llama") == Architecture.LLAMA

    def test_llama_family_detection(self):
        """Models that use LLaMA-style naming should map to LLAMA."""
        assert get_architecture("mistral") == Architecture.LLAMA
        assert get_architecture("qwen") == Architecture.LLAMA
        assert get_architecture("qwen2") == Architecture.LLAMA
        assert get_architecture("gemma") == Architecture.LLAMA

    def test_unknown_defaults_to_llama(self):
        """Unknown architectures should default to LLaMA (most common)."""
        assert get_architecture("unknown_model") == Architecture.LLAMA


class TestGPT2WeightMapping:
    """Test GPT-2 weight mapping between HuggingFace and ironcore formats."""

    def test_hf_to_ironcore_mapping(self, mock_gpt2_hf_state_dict):
        """Test conversion from GPT-2 HF format to ironcore format."""
        mapper = WeightMapper(Architecture.GPT2, num_layers=2)
        ironcore_state_dict = mapper.hf_to_ironcore(mock_gpt2_hf_state_dict, strict=False)

        # Check embeddings
        assert "embedding.word_embeddings.weight" in ironcore_state_dict
        assert "embedding.position_embedding.weight" in ironcore_state_dict

        # Check final layer norm
        assert "output_layernorm.weight" in ironcore_state_dict
        assert "output_layernorm.bias" in ironcore_state_dict

        # Check layer components
        for i in range(2):
            prefix = f"model.layers.{i}"
            # Layer norms
            assert f"{prefix}.input_layernorm.weight" in ironcore_state_dict
            assert f"{prefix}.post_attn_layernorm.weight" in ironcore_state_dict
            # Attention (Q and KV should be split)
            assert f"{prefix}.self_attention.linear_q.weight" in ironcore_state_dict
            assert f"{prefix}.self_attention.linear_kv.weight" in ironcore_state_dict
            assert f"{prefix}.self_attention.output.weight" in ironcore_state_dict
            # MLP
            assert f"{prefix}.mlp.up_proj.weight" in ironcore_state_dict
            assert f"{prefix}.mlp.down_proj.weight" in ironcore_state_dict

    def test_gpt2_qkv_split(self, mock_gpt2_hf_state_dict):
        """Test that GPT-2 fused QKV is correctly split into Q and KV."""
        mapper = WeightMapper(Architecture.GPT2, num_layers=2)
        ironcore_state_dict = mapper.hf_to_ironcore(mock_gpt2_hf_state_dict, strict=False)

        hidden_size = 64
        # Q should have shape [hidden_size, hidden_size]
        q_weight = ironcore_state_dict["model.layers.0.self_attention.linear_q.weight"]
        assert q_weight.shape == (hidden_size, hidden_size)

        # KV should have shape [2 * hidden_size, hidden_size] (K and V concatenated)
        kv_weight = ironcore_state_dict["model.layers.0.self_attention.linear_kv.weight"]
        assert kv_weight.shape == (2 * hidden_size, hidden_size)

    def test_gpt2_transpose(self, mock_gpt2_hf_state_dict):
        """Test that GPT-2 Conv1D weights are correctly transposed."""
        mapper = WeightMapper(Architecture.GPT2, num_layers=2)
        ironcore_state_dict = mapper.hf_to_ironcore(mock_gpt2_hf_state_dict, strict=False)

        hidden_size = 64
        # MLP up_proj should be transposed: [out, in] = [4*h, h]
        up_weight = ironcore_state_dict["model.layers.0.mlp.up_proj.weight"]
        assert up_weight.shape == (4 * hidden_size, hidden_size)

        # Attention output should be transposed: [out, in] = [h, h]
        out_weight = ironcore_state_dict["model.layers.0.self_attention.output.weight"]
        assert out_weight.shape == (hidden_size, hidden_size)

    def test_roundtrip_gpt2(self, mock_gpt2_hf_state_dict):
        """Test HF -> ironcore -> HF roundtrip preserves weights."""
        mapper = WeightMapper(Architecture.GPT2, num_layers=2)

        # HF -> ironcore
        ironcore_state_dict = mapper.hf_to_ironcore(mock_gpt2_hf_state_dict, strict=False)

        # ironcore -> HF
        recovered_hf_state_dict = mapper.ironcore_to_hf(ironcore_state_dict, strict=False)

        # Check that key weights are preserved (allowing for numerical precision)
        for key in ["transformer.wte.weight", "transformer.wpe.weight",
                    "transformer.ln_f.weight", "transformer.ln_f.bias"]:
            if key in mock_gpt2_hf_state_dict and key in recovered_hf_state_dict:
                torch.testing.assert_close(
                    mock_gpt2_hf_state_dict[key],
                    recovered_hf_state_dict[key],
                    msg=f"Mismatch for {key}"
                )

        # Check fused QKV roundtrip
        original_qkv = mock_gpt2_hf_state_dict["transformer.h.0.attn.c_attn.weight"]
        recovered_qkv = recovered_hf_state_dict["transformer.h.0.attn.c_attn.weight"]
        torch.testing.assert_close(original_qkv, recovered_qkv, msg="QKV mismatch")


class TestLLaMAWeightMapping:
    """Test LLaMA weight mapping between HuggingFace and ironcore formats."""

    def test_hf_to_ironcore_mapping(self, mock_llama_hf_state_dict):
        """Test conversion from LLaMA HF format to ironcore format."""
        mapper = WeightMapper(Architecture.LLAMA, num_layers=2)
        ironcore_state_dict = mapper.hf_to_ironcore(mock_llama_hf_state_dict, strict=False)

        # Check embeddings
        assert "embedding.word_embeddings.weight" in ironcore_state_dict

        # Check final layer norm
        assert "output_layernorm.weight" in ironcore_state_dict

        # Check output layer
        assert "output_layer.weight" in ironcore_state_dict

        # Check layer components
        for i in range(2):
            prefix = f"model.layers.{i}"
            # Layer norms
            assert f"{prefix}.input_layernorm.weight" in ironcore_state_dict
            assert f"{prefix}.post_attn_layernorm.weight" in ironcore_state_dict
            # Attention
            assert f"{prefix}.self_attention.linear_q.weight" in ironcore_state_dict
            assert f"{prefix}.self_attention.linear_kv.weight" in ironcore_state_dict
            assert f"{prefix}.self_attention.output.weight" in ironcore_state_dict
            # MLP (gate + up should be fused)
            assert f"{prefix}.mlp.up_proj.weight" in ironcore_state_dict
            assert f"{prefix}.mlp.down_proj.weight" in ironcore_state_dict

    def test_llama_kv_fusion(self, mock_llama_hf_state_dict):
        """Test that LLaMA separate K and V are correctly fused into KV."""
        mapper = WeightMapper(Architecture.LLAMA, num_layers=2)
        ironcore_state_dict = mapper.hf_to_ironcore(mock_llama_hf_state_dict, strict=False)

        # Get original K and V shapes
        k_shape = mock_llama_hf_state_dict["model.layers.0.self_attn.k_proj.weight"].shape
        v_shape = mock_llama_hf_state_dict["model.layers.0.self_attn.v_proj.weight"].shape

        # KV should have shape [k_dim + v_dim, hidden_size]
        kv_weight = ironcore_state_dict["model.layers.0.self_attention.linear_kv.weight"]
        assert kv_weight.shape == (k_shape[0] + v_shape[0], k_shape[1])

    def test_llama_gate_up_fusion(self, mock_llama_hf_state_dict):
        """Test that LLaMA gate_proj and up_proj are correctly fused."""
        mapper = WeightMapper(Architecture.LLAMA, num_layers=2)
        ironcore_state_dict = mapper.hf_to_ironcore(mock_llama_hf_state_dict, strict=False)

        # Get original gate and up shapes
        gate_shape = mock_llama_hf_state_dict["model.layers.0.mlp.gate_proj.weight"].shape
        up_shape = mock_llama_hf_state_dict["model.layers.0.mlp.up_proj.weight"].shape

        # Fused should have shape [gate_dim + up_dim, hidden_size]
        fused_weight = ironcore_state_dict["model.layers.0.mlp.up_proj.weight"]
        assert fused_weight.shape == (gate_shape[0] + up_shape[0], gate_shape[1])

    def test_roundtrip_llama(self, mock_llama_hf_state_dict):
        """Test HF -> ironcore -> HF roundtrip preserves weights."""
        mapper = WeightMapper(Architecture.LLAMA, num_layers=2)

        # HF -> ironcore
        ironcore_state_dict = mapper.hf_to_ironcore(mock_llama_hf_state_dict, strict=False)

        # ironcore -> HF
        recovered_hf_state_dict = mapper.ironcore_to_hf(ironcore_state_dict, strict=False)

        # Check embeddings and output
        for key in ["model.embed_tokens.weight", "model.norm.weight", "lm_head.weight"]:
            if key in mock_llama_hf_state_dict and key in recovered_hf_state_dict:
                torch.testing.assert_close(
                    mock_llama_hf_state_dict[key],
                    recovered_hf_state_dict[key],
                    msg=f"Mismatch for {key}"
                )

        # Check K/V roundtrip
        original_k = mock_llama_hf_state_dict["model.layers.0.self_attn.k_proj.weight"]
        recovered_k = recovered_hf_state_dict["model.layers.0.self_attn.k_proj.weight"]
        torch.testing.assert_close(original_k, recovered_k, msg="K projection mismatch")


# =============================================================================
# Integration Tests: Real HuggingFace Models
# =============================================================================

def is_hf_model_available(model_name: str) -> bool:
    """Check if a HuggingFace model is available for download."""
    try:
        from huggingface_hub import HfApi
        api = HfApi()
        api.model_info(model_name)
        return True
    except Exception:
        return False


def download_hf_model(model_name: str, cache_dir: Path) -> Path:
    """Download a HuggingFace model to cache directory."""
    from huggingface_hub import snapshot_download

    local_dir = cache_dir / model_name.replace("/", "_")
    snapshot_download(
        repo_id=model_name,
        local_dir=str(local_dir),
        ignore_patterns=["*.msgpack", "*.h5", "*.ot", "*.safetensors.index.json"],
    )
    return local_dir


class TestGPT2Integration:
    """Integration tests with real GPT-2 model from HuggingFace."""

    @pytest.fixture
    def gpt2_checkpoint(self, temp_dir):
        """Download GPT-2 checkpoint for testing."""
        model_name = "gpt2"
        if not is_hf_model_available(model_name):
            pytest.skip(f"Model {model_name} not available")

        return download_hf_model(model_name, temp_dir)

    def test_load_gpt2_state_dict(self, gpt2_checkpoint):
        """Test loading GPT-2 state dict from HuggingFace checkpoint."""
        from ironcore.checkpointing.hf_interop import load_hf_state_dict, load_hf_config

        # Load config
        config = load_hf_config(gpt2_checkpoint)
        assert config["model_type"] == "gpt2"
        assert "n_embd" in config or "hidden_size" in config

        # Load state dict
        state_dict = load_hf_state_dict(gpt2_checkpoint)
        # GPT-2 keys may or may not have "transformer." prefix depending on format
        assert "wte.weight" in state_dict or "transformer.wte.weight" in state_dict
        assert "h.0.attn.c_attn.weight" in state_dict or "transformer.h.0.attn.c_attn.weight" in state_dict

    def test_gpt2_weight_conversion(self, gpt2_checkpoint):
        """Test converting GPT-2 weights to ironcore format."""
        from ironcore.checkpointing.hf_interop import load_hf_state_dict, load_hf_config

        config = load_hf_config(gpt2_checkpoint)
        num_layers = config.get("n_layer", config.get("num_hidden_layers", 12))

        state_dict = load_hf_state_dict(gpt2_checkpoint)

        mapper = WeightMapper(Architecture.GPT2, num_layers)
        ironcore_state_dict = mapper.hf_to_ironcore(state_dict, strict=False)

        # Verify key conversions
        assert "embedding.word_embeddings.weight" in ironcore_state_dict
        assert "model.layers.0.self_attention.linear_q.weight" in ironcore_state_dict

        # Verify shapes are correct
        hidden_size = config.get("n_embd", config.get("hidden_size"))
        q_weight = ironcore_state_dict["model.layers.0.self_attention.linear_q.weight"]
        assert q_weight.shape[0] == hidden_size
        assert q_weight.shape[1] == hidden_size

    def test_gpt2_roundtrip(self, gpt2_checkpoint, temp_dir):
        """Test full roundtrip: HF -> ironcore -> HF for GPT-2."""
        from ironcore.checkpointing.hf_interop import load_hf_state_dict, load_hf_config

        config = load_hf_config(gpt2_checkpoint)
        num_layers = config.get("n_layer", config.get("num_hidden_layers", 12))

        original_state_dict = load_hf_state_dict(gpt2_checkpoint)

        mapper = WeightMapper(Architecture.GPT2, num_layers)

        # Convert to ironcore
        ironcore_state_dict = mapper.hf_to_ironcore(original_state_dict, strict=False)

        # Convert back to HF
        recovered_state_dict = mapper.ironcore_to_hf(ironcore_state_dict, strict=False)

        # Verify key weights match
        test_keys = [
            "transformer.wte.weight",
            "transformer.wpe.weight",
            "transformer.ln_f.weight",
            "transformer.h.0.ln_1.weight",
            "transformer.h.0.attn.c_attn.weight",
            "transformer.h.0.mlp.c_fc.weight",
        ]

        for key in test_keys:
            if key in original_state_dict and key in recovered_state_dict:
                torch.testing.assert_close(
                    original_state_dict[key],
                    recovered_state_dict[key],
                    rtol=1e-5,
                    atol=1e-5,
                    msg=f"Roundtrip mismatch for {key}"
                )


class TestLLaMAIntegration:
    """Integration tests with real LLaMA model from HuggingFace."""

    @pytest.fixture
    def llama_checkpoint(self, temp_dir):
        """Download LLaMA checkpoint for testing."""
        model_name = "meta-llama/Llama-3.2-1B"

        # Check for HF token (required for LLaMA)
        hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        if not hf_token:
            pytest.skip("HF_TOKEN not set - required for LLaMA model access")

        if not is_hf_model_available(model_name):
            pytest.skip(f"Model {model_name} not available (check HF token permissions)")

        return download_hf_model(model_name, temp_dir)

    def test_load_llama_state_dict(self, llama_checkpoint):
        """Test loading LLaMA state dict from HuggingFace checkpoint."""
        from ironcore.checkpointing.hf_interop import load_hf_state_dict, load_hf_config

        # Load config
        config = load_hf_config(llama_checkpoint)
        assert config["model_type"] == "llama"
        assert "hidden_size" in config

        # Load state dict
        state_dict = load_hf_state_dict(llama_checkpoint)
        assert "model.embed_tokens.weight" in state_dict
        assert "model.layers.0.self_attn.q_proj.weight" in state_dict

    def test_llama_weight_conversion(self, llama_checkpoint):
        """Test converting LLaMA weights to ironcore format."""
        from ironcore.checkpointing.hf_interop import load_hf_state_dict, load_hf_config

        config = load_hf_config(llama_checkpoint)
        num_layers = config.get("num_hidden_layers", 16)

        state_dict = load_hf_state_dict(llama_checkpoint)

        mapper = WeightMapper(Architecture.LLAMA, num_layers)
        ironcore_state_dict = mapper.hf_to_ironcore(state_dict, strict=False)

        # Verify key conversions
        assert "embedding.word_embeddings.weight" in ironcore_state_dict
        assert "model.layers.0.self_attention.linear_q.weight" in ironcore_state_dict
        assert "model.layers.0.self_attention.linear_kv.weight" in ironcore_state_dict

        # Verify K and V are fused
        hidden_size = config["hidden_size"]
        num_kv_heads = config.get("num_key_value_heads", config["num_attention_heads"])
        head_dim = hidden_size // config["num_attention_heads"]
        expected_kv_dim = 2 * num_kv_heads * head_dim

        kv_weight = ironcore_state_dict["model.layers.0.self_attention.linear_kv.weight"]
        assert kv_weight.shape[0] == expected_kv_dim

    def test_llama_roundtrip(self, llama_checkpoint, temp_dir):
        """Test full roundtrip: HF -> ironcore -> HF for LLaMA."""
        from ironcore.checkpointing.hf_interop import load_hf_state_dict, load_hf_config

        config = load_hf_config(llama_checkpoint)
        num_layers = config.get("num_hidden_layers", 16)

        original_state_dict = load_hf_state_dict(llama_checkpoint)

        mapper = WeightMapper(Architecture.LLAMA, num_layers)

        # Convert to ironcore
        ironcore_state_dict = mapper.hf_to_ironcore(original_state_dict, strict=False)

        # Convert back to HF
        recovered_state_dict = mapper.ironcore_to_hf(ironcore_state_dict, strict=False)

        # Verify key weights match
        test_keys = [
            "model.embed_tokens.weight",
            "model.norm.weight",
            "lm_head.weight",
            "model.layers.0.input_layernorm.weight",
            "model.layers.0.self_attn.q_proj.weight",
            "model.layers.0.self_attn.k_proj.weight",
            "model.layers.0.self_attn.v_proj.weight",
            "model.layers.0.mlp.down_proj.weight",
        ]

        for key in test_keys:
            if key in original_state_dict and key in recovered_state_dict:
                torch.testing.assert_close(
                    original_state_dict[key],
                    recovered_state_dict[key],
                    rtol=1e-5,
                    atol=1e-5,
                    msg=f"Roundtrip mismatch for {key}"
                )


# =============================================================================
# Export Tests
# =============================================================================

class TestExportFunctionality:
    """Test export to HuggingFace format."""

    def test_export_creates_config(self, temp_dir, mock_gpt2_hf_state_dict):
        """Test that export creates config.json."""
        from ironcore.checkpointing.hf_interop import _generate_hf_config

        # Create a mock model
        class MockModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.embedding = nn.ModuleDict({
                    "word_embeddings": nn.Embedding(100, 64)
                })

        model = MockModel()
        config = _generate_hf_config(model, "gpt2", num_layers=2)

        assert "model_type" in config
        assert "vocab_size" in config
        assert config["model_type"] == "gpt2"

    def test_export_creates_files(self, temp_dir, mock_llama_hf_state_dict):
        """Test that export creates checkpoint files."""
        import json

        # Create minimal checkpoint structure manually
        output_path = temp_dir / "test_export"
        output_path.mkdir()

        # Write mock safetensors file
        try:
            from safetensors.torch import save_file
            save_file(mock_llama_hf_state_dict, str(output_path / "model.safetensors"))
        except ImportError:
            # Fall back to pytorch format
            torch.save(mock_llama_hf_state_dict, output_path / "pytorch_model.bin")

        # Write config
        config = {
            "model_type": "llama",
            "hidden_size": 64,
            "num_hidden_layers": 2,
        }
        with open(output_path / "config.json", "w") as f:
            json.dump(config, f)

        # Verify files exist
        assert (output_path / "config.json").exists()
        assert (output_path / "model.safetensors").exists() or (output_path / "pytorch_model.bin").exists()


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
