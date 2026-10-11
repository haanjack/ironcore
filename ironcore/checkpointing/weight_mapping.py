# Copyright (c) 2025-2026 Jaegeun Han
#
# SPDX-License-Identifier: Apache-2.0

"""
Weight mapping utilities for HuggingFace checkpoint interoperability.

This module provides bidirectional mapping between ironcore's internal naming
convention and HuggingFace model naming conventions.

Supported architectures:
- GPT-2 (OpenAI style)
- LLaMA family (LLaMA, LLaMA-2, LLaMA-3, Mistral, Qwen2, Qwen3, etc.)
"""

import re
from enum import Enum

import torch


class Architecture(Enum):
    """Supported HuggingFace model architectures."""

    GPT2 = "gpt2"
    LLAMA = "llama"
    GEMMA4 = "gemma4_text"
    GRANITEMOE = "granitemoe"


# Architecture aliases - many models use LLaMA-style naming
ARCHITECTURE_ALIASES = {
    "granitemoe": Architecture.GRANITEMOE,
    "gemma4": Architecture.GEMMA4,
    "gemma4text": Architecture.GEMMA4,
    "llama": Architecture.LLAMA,
    "llama2": Architecture.LLAMA,
    "llama3": Architecture.LLAMA,
    "mistral": Architecture.LLAMA,  # Same naming as LLaMA
    "mixtral": Architecture.LLAMA,
    "qwen": Architecture.LLAMA,
    "qwen2": Architecture.LLAMA,
    "qwen3": Architecture.LLAMA,
    "gemma": Architecture.LLAMA,
    "gemma2": Architecture.LLAMA,
    # "phi3": Architecture.LLAMA,  # Close enough, with minor differences
    "gpt2": Architecture.GPT2,
    "gpt": Architecture.GPT2,
}


def get_architecture(model_type: str) -> Architecture:
    """Get architecture enum from model type string."""
    model_type_lower = model_type.lower().replace("-", "").replace("_", "")
    return ARCHITECTURE_ALIASES.get(model_type_lower, Architecture.LLAMA)


# =============================================================================
# Ironcore naming convention:
# =============================================================================
# embedding.word_embeddings.weight              - word embedding
# embedding.position_embedding.weight           - absolute position embedding
# model.layers.{i}.input_layernorm.layernorm.weight   - pre-attention layer norm (wrapped)
# model.layers.{i}.linear_q.weight              - query projection (direct on layer)
# model.layers.{i}.linear_kv.weight             - key-value projection (fused, direct on layer)
# model.layers.{i}.attn_output.weight           - attention output (direct on layer)
# model.layers.{i}.post_attn_layernorm.layernorm.weight - post-attention layer norm (wrapped)
# model.layers.{i}.mlp.up_proj.weight           - MLP up projection
# model.layers.{i}.mlp.down_proj.weight         - MLP down projection
# output_layernorm.layernorm.weight             - final layer norm (wrapped)
# output_layer.weight                           - output projection (untied)


# =============================================================================
# GPT-2 HuggingFace naming convention:
# =============================================================================
# transformer.wte.weight                    - word embedding
# transformer.wpe.weight                    - position embedding
# transformer.h.{i}.ln_1.weight/bias        - pre-attention layer norm
# transformer.h.{i}.attn.c_attn.weight/bias - fused QKV (transposed!)
# transformer.h.{i}.attn.c_proj.weight/bias - attention output (transposed!)
# transformer.h.{i}.ln_2.weight/bias        - post-attention layer norm
# transformer.h.{i}.mlp.c_fc.weight/bias    - MLP up (transposed!)
# transformer.h.{i}.mlp.c_proj.weight/bias  - MLP down (transposed!)
# transformer.ln_f.weight/bias              - final layer norm
# lm_head.weight                            - output projection


# =============================================================================
# LLaMA HuggingFace naming convention:
# =============================================================================
# model.embed_tokens.weight                         - word embedding
# model.layers.{i}.input_layernorm.weight           - pre-attention RMSNorm
# model.layers.{i}.self_attn.q_proj.weight          - query
# model.layers.{i}.self_attn.k_proj.weight          - key
# model.layers.{i}.self_attn.v_proj.weight          - value
# model.layers.{i}.self_attn.o_proj.weight          - attention output
# model.layers.{i}.post_attention_layernorm.weight  - post-attention RMSNorm
# model.layers.{i}.mlp.gate_proj.weight             - MLP gate (for SwiGLU)
# model.layers.{i}.mlp.up_proj.weight               - MLP up
# model.layers.{i}.mlp.down_proj.weight             - MLP down
# model.norm.weight                                 - final RMSNorm
# lm_head.weight                                    - output projection


class WeightMapper:
    """
    Handles bidirectional weight mapping between HuggingFace and ironcore formats.

    This class supports:
    - Key name translation
    - Tensor transformations (e.g., transpose for GPT-2 Conv1D weights)
    - Fused/split weight handling (e.g., separate Q/K/V vs fused QKV)
    """

    def __init__(self, architecture: Architecture, num_layers: int):
        self.architecture = architecture
        self.num_layers = num_layers

    def hf_to_ironcore(
        self,
        hf_state_dict: dict[str, torch.Tensor],
        strict: bool = True,
    ) -> dict[str, torch.Tensor]:
        """
        Convert HuggingFace state dict to ironcore format.

        Args:
            hf_state_dict: State dict from HuggingFace checkpoint
            strict: If True, raise error for unmapped keys

        Returns:
            State dict with ironcore naming convention
        """

        if self.architecture == Architecture.GPT2:
            return self._hf_gpt2_to_ironcore(hf_state_dict, strict)
        elif self.architecture == Architecture.LLAMA:
            return self._hf_llama_to_ironcore(hf_state_dict, strict)
        elif self.architecture == Architecture.GEMMA4:
            return self._map_gemma4(hf_state_dict, to_hf=False, strict=strict)
        elif self.architecture == Architecture.GRANITEMOE:
            return self._map_granitemoe(hf_state_dict, to_hf=False, strict=strict)
        else:
            raise ValueError(f"Unsupported architecture: {self.architecture}")

    def ironcore_to_hf(
        self,
        ironcore_state_dict: dict[str, torch.Tensor],
        strict: bool = True,
    ) -> dict[str, torch.Tensor]:
        """
        Convert ironcore state dict to HuggingFace format.

        Args:
            ironcore_state_dict: State dict from ironcore model
            strict: If True, raise error for unmapped keys

        Returns:
            State dict with HuggingFace naming convention
        """

        if self.architecture == Architecture.GPT2:
            return self._ironcore_to_hf_gpt2(ironcore_state_dict, strict)
        elif self.architecture == Architecture.LLAMA:
            return self._ironcore_to_hf_llama(ironcore_state_dict, strict)
        elif self.architecture == Architecture.GEMMA4:
            return self._map_gemma4(ironcore_state_dict, to_hf=True, strict=strict)
        elif self.architecture == Architecture.GRANITEMOE:
            return self._map_granitemoe(ironcore_state_dict, to_hf=True, strict=strict)
        else:
            raise ValueError(f"Unsupported architecture: {self.architecture}")

    def _map_granitemoe(self, state_dict, *, to_hf, strict):
        """Accept both legacy IBM packed keys and current Transformers keys."""
        ordinary, mapped = {}, {}
        packed = {}
        for key, tensor in state_dict.items():
            if to_hf:
                if key == "rotary_pos_emb.theta":
                    continue  # Derived from the serialized RoPE configuration.
                if "lora_" in key or ".base_layer." in key:
                    raise ValueError("Merge Granite LoRA before exporting dense HF weights")
                match = re.fullmatch(
                    r"model.layers.(\d+).mlp.routed_experts.(\d+).(up_proj|down_proj).weight", key
                )
                if match:
                    layer, expert, projection = match.groups()
                    name = "gate_up_proj" if projection == "up_proj" else "down_proj"
                    target = f"model.layers.{layer}.block_sparse_moe.experts.{name}"
                    packed.setdefault(target, {})[int(expert)] = tensor.T
                    continue
                match = re.fullmatch(r"model.layers.(\d+).mlp.router.weight", key)
                if match:
                    mapped[f"model.layers.{match[1]}.block_sparse_moe.router.weight"] = tensor.T
                    continue
            else:
                match = re.fullmatch(
                    r"model.layers.(\d+).block_sparse_moe.(input_linear.weight|output_linear.weight|"
                    r"experts.gate_up_proj|experts.down_proj|router.layer.weight|router.weight)",
                    key,
                )
                if match:
                    layer, name = match.groups()
                    prefix = f"model.layers.{layer}.mlp"
                    if name.startswith("router"):
                        mapped[f"{prefix}.router.weight"] = tensor.T
                    else:
                        projection = (
                            "up_proj"
                            if name in {"input_linear.weight", "experts.gate_up_proj"}
                            else "down_proj"
                        )
                        for expert, value in enumerate(tensor):
                            mapped[f"{prefix}.routed_experts.{expert}.{projection}.weight"] = (
                                value.T
                            )
                    continue
            ordinary[key] = tensor
        for key, experts in packed.items():
            if sorted(experts) != list(range(len(experts))):
                raise ValueError(f"Noncontiguous Granite expert indices for {key}")
            mapped[key] = torch.stack([experts[i] for i in range(len(experts))])
        mapper = self._ironcore_to_hf_llama if to_hf else self._hf_llama_to_ironcore
        mapped.update(mapper(ordinary, strict))
        return mapped

    @staticmethod
    def _map_gemma4(
        state_dict: dict[str, torch.Tensor], *, to_hf: bool, strict: bool
    ) -> dict[str, torch.Tensor]:
        """Map dense text weights without collapsing Gemma 4 into LLaMA.

        Public multimodal checkpoints prefix text weights with
        model.language_model.; exported checkpoints are Gemma4ForCausalLM.
        """
        output = {}
        expert_groups = {}
        projections = (
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
            "per_layer_input_gate",
            "per_layer_projection",
            "per_layer_model_projection",
        )
        for raw_name, tensor in state_dict.items():
            name = raw_name
            if to_hf and (
                ".base_layer." in name or ".lora." in name or name.endswith((".lora_A", ".lora_B"))
            ):
                raise ValueError(
                    "Merge Gemma 4 LoRA adapters before exporting a dense HF checkpoint"
                )
            if not to_hf:
                for prefix in ("model.language_model.", "language_model."):
                    if name.startswith(prefix):
                        name = "model." + name[len(prefix) :]
                        break
            specials = (
                {
                    "embedding.word_embeddings.weight": "model.embed_tokens.weight",
                    "output_layernorm.weight": "model.norm.weight",
                }
                if to_hf
                else {
                    "model.embed_tokens.weight": "embedding.word_embeddings.weight",
                    "model.norm.weight": "output_layernorm.weight",
                }
            )
            expert_match = (
                re.fullmatch(r"(model\.layers\.\d+\.experts)\.(gate_up_proj|down_proj)", name)
                if not to_hf
                else None
            )
            export_match = (
                re.fullmatch(
                    r"(model\.layers\.\d+\.experts)\.(\d+)\.(up_proj|down_proj)\.weight", name
                )
                if to_hf
                else None
            )
            if expert_match:
                prefix, projection = expert_match.groups()
                native_projection = "up_proj" if projection == "gate_up_proj" else "down_proj"
                for index, expert in enumerate(tensor):
                    # Import copies into native parameters; keep source views
                    # instead of allocating all transposed experts at once.
                    output[f"{prefix}.{index}.{native_projection}.weight"] = expert.t()
            elif export_match:
                prefix, index, projection = export_match.groups()
                key = f"{prefix}.{'gate_up_proj' if projection == 'up_proj' else 'down_proj'}"
                expert_groups.setdefault(key, {})[int(index)] = tensor.t().contiguous()
            elif name in specials:
                output[specials[name]] = tensor
            elif not to_hf and name == "lm_head.weight":
                # Dense Gemma 4 ties its head to the token embedding.
                continue
            elif name.startswith(
                (
                    "model.layers.",
                    "model.embed_tokens_per_layer.",
                    "model.per_layer_model_projection.",
                    "model.per_layer_projection_norm.",
                )
            ):
                if name.endswith(".weight") and any(
                    name.endswith(f".{projection}.weight") for projection in projections
                ):
                    tensor = tensor.t().contiguous()
                output[name] = tensor
            elif strict:
                raise ValueError(f"Unmapped Gemma 4 key: {raw_name}")
        if to_hf and "model.embed_tokens.weight" in output:
            output["lm_head.weight"] = output["model.embed_tokens.weight"].clone()
        for key, experts in expert_groups.items():
            output[key] = torch.stack([experts[i] for i in range(len(experts))])
        return output

    # =========================================================================
    # GPT-2 Conversion
    # =========================================================================

    def _hf_gpt2_to_ironcore(
        self,
        hf_state_dict: dict[str, torch.Tensor],
        strict: bool = True,
    ) -> dict[str, torch.Tensor]:
        """Convert GPT-2 HuggingFace checkpoint to ironcore format."""

        ironcore_state_dict = {}
        mapped_keys = set()

        for hf_key, tensor in hf_state_dict.items():
            ironcore_key, transformed_tensor = self._map_gpt2_key_to_ironcore(hf_key, tensor)

            if ironcore_key is not None:
                if isinstance(ironcore_key, tuple):
                    # Multiple outputs (e.g., split QKV)
                    for k, t in zip(ironcore_key, transformed_tensor, strict=True):
                        ironcore_state_dict[k] = t
                else:
                    ironcore_state_dict[ironcore_key] = transformed_tensor
                mapped_keys.add(hf_key)

        if strict:
            unmapped = set(hf_state_dict.keys()) - mapped_keys
            # Filter out keys that are expected to be unmapped
            unmapped = {k for k in unmapped if not self._is_ignorable_key(k)}
            if unmapped:
                raise ValueError(f"Unmapped HuggingFace keys: {unmapped}")

        return ironcore_state_dict

    def _map_gpt2_key_to_ironcore(
        self,
        hf_key: str,
        tensor: torch.Tensor,
    ) -> tuple[str | tuple[str, ...] | None, torch.Tensor | tuple[torch.Tensor, ...] | None]:
        """Map a single GPT-2 HuggingFace key to ironcore format."""

        # Normalize key - HF GPT-2 may or may not have "transformer." prefix
        # depending on how it was saved (safetensors vs pytorch_model.bin)
        normalized_key = hf_key
        if not hf_key.startswith("transformer.") and not hf_key.startswith("lm_head"):
            # Add prefix if missing (safetensors format)
            if (
                hf_key.startswith("wte.")
                or hf_key.startswith("wpe.")
                or hf_key.startswith("ln_f.")
                or hf_key.startswith("h.")
            ):
                normalized_key = "transformer." + hf_key

        # Simple non-layer mappings
        # Note: ironcore's LayerNorm wraps nn.LayerNorm, so weights have .layernorm suffix
        simple_mappings = {
            "transformer.wte.weight": "embedding.word_embeddings.weight",
            "transformer.wpe.weight": "embedding.position_embedding.weight",
            "transformer.ln_f.weight": "output_layernorm.layernorm.weight",
            "transformer.ln_f.bias": "output_layernorm.layernorm.bias",
            "lm_head.weight": "output_layer.weight",
        }

        if normalized_key in simple_mappings:
            return simple_mappings[
                normalized_key
            ], tensor.T if normalized_key == "lm_head.weight" else tensor

        # Layer-specific mappings
        layer_match = re.match(r"transformer\.h\.(\d+)\.(.*)", normalized_key)
        if layer_match:
            layer_idx = layer_match.group(1)
            layer_key = layer_match.group(2)

            # Attention QKV (fused in GPT-2, need to split for ironcore)
            # Both GPT-2 Conv1D and ironcore ParallelLinear use: y = x @ W
            # So weights have shape [in_features, out_features] - NO transpose needed
            if layer_key == "attn.c_attn.weight":
                # GPT-2 c_attn: [hidden_size, 3 * hidden_size] (Conv1D style)
                # Split into Q, KV without transposing
                hidden_size = tensor.shape[0]
                q, k, v = tensor.split(hidden_size, dim=1)  # Split along output dim
                kv = torch.cat([k, v], dim=1)  # Fuse K and V along output dim
                return (
                    (
                        f"model.layers.{layer_idx}.linear_q.weight",
                        f"model.layers.{layer_idx}.linear_kv.weight",
                    ),
                    (q, kv),
                )
            if layer_key == "attn.c_attn.bias":
                hidden_size = tensor.shape[0] // 3
                q, k, v = tensor.split(hidden_size, dim=0)
                kv = torch.cat([k, v], dim=0)
                return (
                    (
                        f"model.layers.{layer_idx}.linear_q.bias",
                        f"model.layers.{layer_idx}.linear_kv.bias",
                    ),
                    (q, kv),
                )

            # Layer mappings - NO transformation needed
            # Both GPT-2 Conv1D and ironcore ParallelLinear use: y = x @ W
            # So weights have shape [in_features, out_features] - NO transpose needed
            transform_mappings = {
                "attn.c_proj.weight": f"model.layers.{layer_idx}.attn_output.weight",
                "mlp.c_fc.weight": f"model.layers.{layer_idx}.mlp.up_proj.weight",
                "mlp.c_proj.weight": f"model.layers.{layer_idx}.mlp.down_proj.weight",
            }

            if layer_key in transform_mappings:
                return transform_mappings[layer_key], tensor

            # Simple layer mappings
            # Note: ironcore's LayerNorm wraps nn.LayerNorm, so weights have .layernorm suffix
            layer_simple_mappings = {
                "ln_1.weight": f"model.layers.{layer_idx}.input_layernorm.layernorm.weight",
                "ln_1.bias": f"model.layers.{layer_idx}.input_layernorm.layernorm.bias",
                "attn.c_proj.bias": f"model.layers.{layer_idx}.attn_output.bias",
                "ln_2.weight": f"model.layers.{layer_idx}.post_attn_layernorm.layernorm.weight",
                "ln_2.bias": f"model.layers.{layer_idx}.post_attn_layernorm.layernorm.bias",
                "mlp.c_fc.bias": f"model.layers.{layer_idx}.mlp.up_proj.bias",
                "mlp.c_proj.bias": f"model.layers.{layer_idx}.mlp.down_proj.bias",
            }

            if layer_key in layer_simple_mappings:
                return layer_simple_mappings[layer_key], tensor

        return None, None

    def _ironcore_to_hf_gpt2(
        self,
        ironcore_state_dict: dict[str, torch.Tensor],
        strict: bool = True,
    ) -> dict[str, torch.Tensor]:
        """Convert ironcore checkpoint to GPT-2 HuggingFace format."""
        import torch

        hf_state_dict = {}
        mapped_keys = set()

        # Process non-layer keys first
        # Note: ironcore's LayerNorm wraps nn.LayerNorm, so weights have .layernorm suffix
        simple_mappings = {
            "embedding.word_embeddings.weight": "transformer.wte.weight",
            "embedding.position_embedding.weight": "transformer.wpe.weight",
            "output_layernorm.layernorm.weight": "transformer.ln_f.weight",
            "output_layernorm.layernorm.bias": "transformer.ln_f.bias",
            "output_layer.weight": "lm_head.weight",
        }

        for ic_key, hf_key in simple_mappings.items():
            if ic_key in ironcore_state_dict:
                value = ironcore_state_dict[ic_key]
                hf_state_dict[hf_key] = value.T if ic_key == "output_layer.weight" else value
                mapped_keys.add(ic_key)

        # Process layer keys
        for layer_idx in range(self.num_layers):
            prefix = f"model.layers.{layer_idx}"
            hf_prefix = f"transformer.h.{layer_idx}"

            # Layer norms (ironcore wraps nn.LayerNorm, so weights have .layernorm suffix)
            for ic_suffix, hf_suffix in [
                ("input_layernorm.layernorm.weight", "ln_1.weight"),
                ("input_layernorm.layernorm.bias", "ln_1.bias"),
                ("post_attn_layernorm.layernorm.weight", "ln_2.weight"),
                ("post_attn_layernorm.layernorm.bias", "ln_2.bias"),
            ]:
                ic_key = f"{prefix}.{ic_suffix}"
                if ic_key in ironcore_state_dict:
                    hf_state_dict[f"{hf_prefix}.{hf_suffix}"] = ironcore_state_dict[ic_key]
                    mapped_keys.add(ic_key)

            # Fuse Q and KV back to c_attn
            # Both use same convention: [in_features, out_features] - NO transpose needed
            q_key = f"{prefix}.linear_q.weight"
            kv_key = f"{prefix}.linear_kv.weight"
            if q_key in ironcore_state_dict and kv_key in ironcore_state_dict:
                q = ironcore_state_dict[q_key]
                kv = ironcore_state_dict[kv_key]
                k, v = kv.chunk(2, dim=1)  # Split along output dim
                # GPT-2 expects [hidden_size, 3 * hidden_size] - same convention
                c_attn = torch.cat([q, k, v], dim=1)
                hf_state_dict[f"{hf_prefix}.attn.c_attn.weight"] = c_attn
                mapped_keys.add(q_key)
                mapped_keys.add(kv_key)

            q_bias_key = f"{prefix}.linear_q.bias"
            kv_bias_key = f"{prefix}.linear_kv.bias"
            if q_bias_key in ironcore_state_dict and kv_bias_key in ironcore_state_dict:
                q_bias = ironcore_state_dict[q_bias_key]
                kv_bias = ironcore_state_dict[kv_bias_key]
                k_bias, v_bias = kv_bias.chunk(2, dim=0)
                c_attn_bias = torch.cat([q_bias, k_bias, v_bias], dim=0)
                hf_state_dict[f"{hf_prefix}.attn.c_attn.bias"] = c_attn_bias
                mapped_keys.add(q_bias_key)
                mapped_keys.add(kv_bias_key)

            # Attention output - NO transpose needed
            out_key = f"{prefix}.attn_output.weight"
            if out_key in ironcore_state_dict:
                hf_state_dict[f"{hf_prefix}.attn.c_proj.weight"] = ironcore_state_dict[out_key]
                mapped_keys.add(out_key)
            out_bias_key = f"{prefix}.attn_output.bias"
            if out_bias_key in ironcore_state_dict:
                hf_state_dict[f"{hf_prefix}.attn.c_proj.bias"] = ironcore_state_dict[out_bias_key]
                mapped_keys.add(out_bias_key)

            # MLP - NO transpose needed (same convention)
            for ic_suffix, hf_suffix in [
                ("mlp.up_proj.weight", "mlp.c_fc.weight"),
                ("mlp.up_proj.bias", "mlp.c_fc.bias"),
                ("mlp.down_proj.weight", "mlp.c_proj.weight"),
                ("mlp.down_proj.bias", "mlp.c_proj.bias"),
            ]:
                ic_key = f"{prefix}.{ic_suffix}"
                if ic_key in ironcore_state_dict:
                    hf_state_dict[f"{hf_prefix}.{hf_suffix}"] = ironcore_state_dict[ic_key]
                    mapped_keys.add(ic_key)

        return hf_state_dict

    # =========================================================================
    # LLaMA Conversion
    # =========================================================================

    def _hf_llama_to_ironcore(
        self,
        hf_state_dict: dict[str, torch.Tensor],
        strict: bool = True,
    ) -> dict[str, torch.Tensor]:
        """Convert LLaMA HuggingFace checkpoint to ironcore format."""
        import torch

        ironcore_state_dict = {}
        mapped_keys = set()

        for hf_key, tensor in hf_state_dict.items():
            ironcore_key, transformed_tensor = self._map_llama_key_to_ironcore(
                hf_key, tensor, hf_state_dict
            )

            if ironcore_key is not None:
                ironcore_state_dict[ironcore_key] = transformed_tensor
                mapped_keys.add(hf_key)

        # Handle K/V fusion (LLaMA has separate K and V, ironcore uses fused KV)
        for layer_idx in range(self.num_layers):
            k_key = f"model.layers.{layer_idx}.self_attn.k_proj.weight"
            v_key = f"model.layers.{layer_idx}.self_attn.v_proj.weight"

            if k_key in hf_state_dict and v_key in hf_state_dict:
                k = hf_state_dict[k_key]  # HF: [gn*hd, hidden]
                v = hf_state_dict[v_key]  # HF: [gn*hd, hidden]
                # Fuse along out dim then transpose to IC [in, out] convention
                kv = torch.cat([k, v], dim=0).t()  # → [hidden, 2*gn*hd]
                ironcore_state_dict[f"model.layers.{layer_idx}.linear_kv.weight"] = kv
                mapped_keys.add(k_key)
                mapped_keys.add(v_key)

            # Handle biases if present
            k_bias_key = f"model.layers.{layer_idx}.self_attn.k_proj.bias"
            v_bias_key = f"model.layers.{layer_idx}.self_attn.v_proj.bias"
            if k_bias_key in hf_state_dict and v_bias_key in hf_state_dict:
                k_bias = hf_state_dict[k_bias_key]
                v_bias = hf_state_dict[v_bias_key]
                kv_bias = torch.cat([k_bias, v_bias], dim=0)
                ironcore_state_dict[f"model.layers.{layer_idx}.linear_kv.bias"] = kv_bias
                mapped_keys.add(k_bias_key)
                mapped_keys.add(v_bias_key)

        if strict:
            unmapped = set(hf_state_dict.keys()) - mapped_keys
            unmapped = {k for k in unmapped if not self._is_ignorable_key(k)}
            if unmapped:
                raise ValueError(f"Unmapped HuggingFace keys: {unmapped}")

        return ironcore_state_dict

    def _map_llama_key_to_ironcore(  # noqa: PLR0911
        self,
        hf_key: str,
        tensor: torch.Tensor,
        full_state_dict: dict[str, torch.Tensor],
    ) -> tuple[str | tuple[str, ...] | None, torch.Tensor | tuple[torch.Tensor, ...] | None]:
        """Map a single LLaMA HuggingFace key to ironcore format."""

        # Simple non-layer mappings
        simple_mappings = {
            "model.embed_tokens.weight": "embedding.word_embeddings.weight",
            "model.norm.weight": "output_layernorm.layernorm.weight",
            "model.norm.bias": "output_layernorm.layernorm.bias",
            "lm_head.weight": "output_layer.weight",
        }

        if hf_key in simple_mappings:
            return simple_mappings[hf_key], tensor.T if hf_key == "lm_head.weight" else tensor

        # Layer-specific mappings
        layer_match = re.match(r"model\.layers\.(\d+)\.(.*)", hf_key)
        if not layer_match:
            return None, None

        layer_idx = layer_match.group(1)
        layer_key = layer_match.group(2)

        # K and V are handled separately (fused in _hf_llama_to_ironcore)
        if layer_key in [
            "self_attn.k_proj.weight",
            "self_attn.k_proj.bias",
            "self_attn.v_proj.weight",
            "self_attn.v_proj.bias",
        ]:
            return None, None  # Skip, handled in fusion step

        # MLP - LLaMA uses gate_proj + up_proj (SwiGLU), ironcore fuses them
        if layer_key in ("mlp.gate_proj.weight", "mlp.up_proj.weight"):
            return self._handle_llama_mlp_fusion(layer_idx, layer_key, tensor, full_state_dict)

        # Linear weights: HF stores [out, in], ironcore uses [in, out] — transpose required
        weight_mappings = {
            "self_attn.q_proj.weight": f"model.layers.{layer_idx}.linear_q.weight",
            "self_attn.o_proj.weight": f"model.layers.{layer_idx}.attn_output.weight",
            "mlp.down_proj.weight": f"model.layers.{layer_idx}.mlp.down_proj.weight",
        }
        if layer_key in weight_mappings:
            return weight_mappings[layer_key], tensor.t()

        # Norms and biases: 1D tensors, no transform needed
        no_transform_mappings = {
            "input_layernorm.weight": f"model.layers.{layer_idx}.input_layernorm.layernorm.weight",
            "input_layernorm.bias": f"model.layers.{layer_idx}.input_layernorm.layernorm.bias",
            "self_attn.q_proj.bias": f"model.layers.{layer_idx}.linear_q.bias",
            "self_attn.o_proj.bias": f"model.layers.{layer_idx}.attn_output.bias",
            "post_attention_layernorm.weight": f"model.layers.{layer_idx}.post_attn_layernorm.layernorm.weight",
            "post_attention_layernorm.bias": f"model.layers.{layer_idx}.post_attn_layernorm.layernorm.bias",
            "mlp.down_proj.bias": f"model.layers.{layer_idx}.mlp.down_proj.bias",
        }
        if layer_key in no_transform_mappings:
            return no_transform_mappings[layer_key], tensor

        return None, None

    def _handle_llama_mlp_fusion(
        self,
        layer_idx: str,
        layer_key: str,
        tensor: torch.Tensor,
        full_state_dict: dict[str, torch.Tensor],
    ) -> tuple[str | None, torch.Tensor | None]:
        """Handle LLaMA MLP gate/up projection fusion logic.

        HF stores MLP weights as [out_features, in_features] (transposed).
        IronCore stores them as [in_features, out_features].
        Need to transpose after fusion.
        """
        if layer_key == "mlp.gate_proj.weight":
            # Check if we need to fuse with up_proj
            up_key = f"model.layers.{layer_idx}.mlp.up_proj.weight"
            if up_key in full_state_dict:
                import torch

                gate = tensor  # HF: [intermediate_size, hidden_size]
                up = full_state_dict[up_key]  # HF: [intermediate_size, hidden_size]
                # Fuse gate and up for SwiGLU along output dim: [2*intermediate_size, hidden_size]
                fused = torch.cat([gate, up], dim=0)
                # Transpose to IronCore format: [hidden_size, 2*intermediate_size]
                fused = fused.t()
                return f"model.layers.{layer_idx}.mlp.up_proj.weight", fused
            # Single gate_proj (no fusion) - still need to transpose
            return f"model.layers.{layer_idx}.mlp.gate_proj.weight", tensor.t()

        # layer_key == "mlp.up_proj.weight"
        # Skip if gate_proj exists (handled in gate_proj fusion)
        gate_key = f"model.layers.{layer_idx}.mlp.gate_proj.weight"
        if gate_key in full_state_dict:
            return None, None
        # Single up_proj (no gate) - still need to transpose
        return f"model.layers.{layer_idx}.mlp.up_proj.weight", tensor.t()

    def _ironcore_to_hf_llama(
        self,
        ironcore_state_dict: dict[str, torch.Tensor],
        strict: bool = True,
    ) -> dict[str, torch.Tensor]:
        """Convert ironcore checkpoint to LLaMA HuggingFace format."""

        hf_state_dict = {}
        mapped_keys = set()

        # Simple mappings
        simple_mappings = {
            "embedding.word_embeddings.weight": "model.embed_tokens.weight",
            "output_layernorm.layernorm.weight": "model.norm.weight",
            "output_layernorm.layernorm.bias": "model.norm.bias",
            "output_layer.weight": "lm_head.weight",
        }

        for ic_key, hf_key in simple_mappings.items():
            if ic_key in ironcore_state_dict:
                value = ironcore_state_dict[ic_key]
                hf_state_dict[hf_key] = value.T if ic_key == "output_layer.weight" else value
                mapped_keys.add(ic_key)

        # Process layer keys
        for layer_idx in range(self.num_layers):
            prefix = f"model.layers.{layer_idx}"
            hf_prefix = f"model.layers.{layer_idx}"

            # Layer norms
            for ic_suffix, hf_suffix in [
                ("input_layernorm.layernorm.weight", "input_layernorm.weight"),
                ("input_layernorm.layernorm.bias", "input_layernorm.bias"),
                ("post_attn_layernorm.layernorm.weight", "post_attention_layernorm.weight"),
                ("post_attn_layernorm.layernorm.bias", "post_attention_layernorm.bias"),
            ]:
                ic_key = f"{prefix}.{ic_suffix}"
                if ic_key in ironcore_state_dict:
                    hf_state_dict[f"{hf_prefix}.{hf_suffix}"] = ironcore_state_dict[ic_key]
                    mapped_keys.add(ic_key)

            # Query projection — IC [in, out] → HF [out, in]
            q_key = f"{prefix}.linear_q.weight"
            if q_key in ironcore_state_dict:
                hf_state_dict[f"{hf_prefix}.self_attn.q_proj.weight"] = ironcore_state_dict[
                    q_key
                ].t()
                mapped_keys.add(q_key)
            q_bias_key = f"{prefix}.linear_q.bias"
            if q_bias_key in ironcore_state_dict:
                hf_state_dict[f"{hf_prefix}.self_attn.q_proj.bias"] = ironcore_state_dict[
                    q_bias_key
                ]
                mapped_keys.add(q_bias_key)

            # Split KV back to K and V — IC [hidden, 2*gn*hd] → HF [gn*hd, hidden] each
            kv_key = f"{prefix}.linear_kv.weight"
            if kv_key in ironcore_state_dict:
                kv = ironcore_state_dict[kv_key]  # [hidden, 2*gn*hd]
                k_t, v_t = kv.chunk(2, dim=1)  # [hidden, gn*hd] each
                hf_state_dict[f"{hf_prefix}.self_attn.k_proj.weight"] = k_t.t()
                hf_state_dict[f"{hf_prefix}.self_attn.v_proj.weight"] = v_t.t()
                mapped_keys.add(kv_key)

            kv_bias_key = f"{prefix}.linear_kv.bias"
            if kv_bias_key in ironcore_state_dict:
                kv_bias = ironcore_state_dict[kv_bias_key]
                k_bias, v_bias = kv_bias.chunk(2, dim=0)
                hf_state_dict[f"{hf_prefix}.self_attn.k_proj.bias"] = k_bias
                hf_state_dict[f"{hf_prefix}.self_attn.v_proj.bias"] = v_bias
                mapped_keys.add(kv_bias_key)

            # Attention output — IC [in, out] → HF [out, in]
            out_key = f"{prefix}.attn_output.weight"
            if out_key in ironcore_state_dict:
                hf_state_dict[f"{hf_prefix}.self_attn.o_proj.weight"] = ironcore_state_dict[
                    out_key
                ].t()
                mapped_keys.add(out_key)
            out_bias_key = f"{prefix}.attn_output.bias"
            if out_bias_key in ironcore_state_dict:
                hf_state_dict[f"{hf_prefix}.self_attn.o_proj.bias"] = ironcore_state_dict[
                    out_bias_key
                ]
                mapped_keys.add(out_bias_key)

            # MLP - split fused gate+up back to separate, IC [in, out] → HF [out, in]
            up_key = f"{prefix}.mlp.up_proj.weight"
            if up_key in ironcore_state_dict:
                fused = ironcore_state_dict[
                    up_key
                ]  # IC: [hidden, 2*ffn] if fused, else [hidden, ffn]
                # Fused gate+up: output dim (dim=1) is 2x a single projection
                if fused.shape[1] % 2 == 0:
                    gate_t, up_t = fused.chunk(2, dim=1)  # [hidden, ffn] each
                    hf_state_dict[f"{hf_prefix}.mlp.gate_proj.weight"] = gate_t.t()  # [ffn, hidden]
                    hf_state_dict[f"{hf_prefix}.mlp.up_proj.weight"] = up_t.t()  # [ffn, hidden]
                else:
                    hf_state_dict[f"{hf_prefix}.mlp.up_proj.weight"] = fused.t()
                mapped_keys.add(up_key)

            down_key = f"{prefix}.mlp.down_proj.weight"
            if down_key in ironcore_state_dict:
                # IC [ffn, hidden] → HF [hidden, ffn]
                hf_state_dict[f"{hf_prefix}.mlp.down_proj.weight"] = ironcore_state_dict[
                    down_key
                ].t()
                mapped_keys.add(down_key)
            down_bias_key = f"{prefix}.mlp.down_proj.bias"
            if down_bias_key in ironcore_state_dict:
                hf_state_dict[f"{hf_prefix}.mlp.down_proj.bias"] = ironcore_state_dict[
                    down_bias_key
                ]
                mapped_keys.add(down_bias_key)

        return hf_state_dict

    def _is_ignorable_key(self, key: str) -> bool:
        """Check if a key can be safely ignored during mapping."""
        ignorable_patterns = [
            r".*\.rotary_emb\.inv_freq",  # RoPE frequencies (computed, not learned)
            r".*\.attention\.masked_bias",  # Attention mask bias
            r".*\.attention\.bias",  # Causal mask
        ]
        return any(re.match(pattern, key) for pattern in ignorable_patterns)
