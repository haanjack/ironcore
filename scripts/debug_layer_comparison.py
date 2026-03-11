#!/usr/bin/env python
"""Debug script to compare HF and IronCore layer outputs step by step."""

import sys
import os

# Set up paths
script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(script_dir)
sys.path.insert(0, project_dir)
os.chdir(project_dir)

# Set environment variables for config validation
os.environ['WORLD_SIZE'] = '2'

sys.argv = ['debug', '--config-path', 'configs/grpo_gsm8k_smoke.yaml']

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer as HFAutoTokenizer
from huggingface_hub import snapshot_download

# HF setup
print("=== Loading HF Model ===")
hf_model = AutoModelForCausalLM.from_pretrained(
    'Qwen/Qwen2.5-0.5B-Instruct', torch_dtype=torch.bfloat16, device_map='auto'
)
hf_tokenizer = HFAutoTokenizer.from_pretrained('Qwen/Qwen2.5-0.5B-Instruct')
hf_model.eval()

# IC setup
print("\n=== Loading IronCore Model ===")
from ironcore import get_tokenizer
from ironcore.config import load_trainer_config
from ironcore.global_vars import set_global_states
from ironcore.language_model import LanguageModel
from ironcore.parallel.parallel_states import initialize_model_parallel
from ironcore.checkpointing.hf_interop import load_from_huggingface

config = load_trainer_config()
initialize_model_parallel(1, timeout_in_minutes=10)
set_global_states(config)
ic_model = LanguageModel(config).to('cuda')
cache_dir = snapshot_download(config.trainer.load_from_hf)
load_from_huggingface(cache_dir, ic_model, 'qwen2', strict=False)
ic_model.eval()

# Prepare input
messages = [{'role': 'user', 'content': 'What is 2+2?'}]
hf_enc = hf_tokenizer.apply_chat_template(
    messages, add_generation_prompt=True, return_tensors='pt'
)
if hasattr(hf_enc, 'input_ids'):
    input_ids = hf_enc['input_ids']
elif isinstance(hf_enc, dict):
    input_ids = hf_enc['input_ids']
else:
    input_ids = hf_enc
input_ids = input_ids.to('cuda')

print(f"\nInput IDs: {input_ids}")
print(f"Input shape: {input_ids.shape}")

# Step 1: Compare embeddings
print("\n=== Step 1: Embedding Comparison ===")
with torch.no_grad():
    hf_embed = hf_model.model.embed_tokens(input_ids)
    ic_embed = ic_model.embedding.word_embeddings(input_ids)

embed_diff = (hf_embed.float() - ic_embed.float()).abs()
print(f"HF embed shape: {hf_embed.shape}, dtype: {hf_embed.dtype}")
print(f"IC embed shape: {ic_embed.shape}, dtype: {ic_embed.dtype}")
print(f"Embedding diff: max={embed_diff.max().item():.6f}, mean={embed_diff.mean().item():.6f}")

# Step 2: Compare layernorm outputs
print("\n=== Step 2: Layer 0 Input LayerNorm ===")
with torch.no_grad():
    hf_ln0 = hf_model.model.layers[0].input_layernorm(hf_embed)
    ic_ln0 = ic_model.model.layers[0].input_layernorm(ic_embed)

ln0_diff = (hf_ln0.float() - ic_ln0.float()).abs()
print(f"HF LN0 shape: {hf_ln0.shape}")
print(f"IC LN0 shape: {ic_ln0.shape}")
print(f"LN0 diff: max={ln0_diff.max().item():.6f}, mean={ln0_diff.mean().item():.6f}")

# Compare LN weights
hf_ln_weight = hf_model.model.layers[0].input_layernorm.weight
ic_ln_weight = ic_model.model.layers[0].input_layernorm.layernorm.weight
ln_weight_diff = (hf_ln_weight.float() - ic_ln_weight.float()).abs()
print(f"LN0 weight diff: max={ln_weight_diff.max().item():.6f}")

# Step 3: Compare Q projection outputs
print("\n=== Step 3: Q Projection ===")
with torch.no_grad():
    # HF: projects from normalized hidden states
    hf_q = hf_model.model.layers[0].self_attn.q_proj(hf_ln0)
    # IC: projects from normalized hidden states
    ic_q = ic_model.model.layers[0].linear_q(ic_ln0)

q_diff = (hf_q.float() - ic_q.float()).abs()
print(f"HF Q shape: {hf_q.shape}")
print(f"IC Q shape: {ic_q.shape}")
print(f"Q output diff: max={q_diff.max().item():.6f}, mean={q_diff.mean().item():.6f}")

# Compare Q weights directly
hf_q_weight = hf_model.model.layers[0].self_attn.q_proj.weight
ic_q_weight = ic_model.model.layers[0].linear_q.weight
print(f"HF Q weight shape: {hf_q_weight.shape}")  # [out, in]
print(f"IC Q weight shape: {ic_q_weight.shape}")  # [in, out]
print(f"HF Q weight dtype: {hf_q_weight.dtype}")
print(f"IC Q weight dtype: {ic_q_weight.dtype}")

# Check if transposed weight matches
transposed_diff = (hf_q_weight.float().t() - ic_q_weight.float()).abs()
print(f"Transposed Q weight diff: max={transposed_diff.max().item():.6f}")

# Check Q bias
hf_q_bias = hf_model.model.layers[0].self_attn.q_proj.bias
ic_q_bias = ic_model.model.layers[0].linear_q.bias
if hf_q_bias is not None and ic_q_bias is not None:
    bias_diff = (hf_q_bias.float() - ic_q_bias.float()).abs()
    print(f"Q bias diff: max={bias_diff.max().item():.6f}")
else:
    print(f"HF Q bias: {hf_q_bias}, IC Q bias: {ic_q_bias}")

# Manual matmul to verify
print("\n=== Manual Q MatMul Verification ===")
with torch.no_grad():
    # HF convention: output = input @ weight.T + bias
    # IC convention: output = input @ weight + bias
    manual_hf_q = torch.matmul(hf_ln0, hf_q_weight.t())
    if hf_model.model.layers[0].self_attn.q_proj.bias is not None:
        manual_hf_q = manual_hf_q + hf_q_bias

    manual_ic_q = torch.matmul(ic_ln0, ic_q_weight)
    if ic_model.model.layers[0].linear_q.bias is not None:
        manual_ic_q = manual_ic_q + ic_q_bias

    print(f"Manual HF Q vs actual: max diff = {(manual_hf_q - hf_q).abs().max().item():.6f}")
    print(f"Manual IC Q vs actual: max diff = {(manual_ic_q - ic_q).abs().max().item():.6f}")

    # Cross-check: HF LN0 @ IC weight
    cross_q = torch.matmul(hf_ln0, ic_q_weight)
    if ic_model.model.layers[0].linear_q.bias is not None:
        cross_q = cross_q + ic_q_bias
    cross_diff = (cross_q - hf_q).abs()
    print(f"HF LN0 @ IC Q weight vs HF Q: max diff = {cross_diff.max().item():.6f}")

# Step 4: Check full layer 0 output
print("\n=== Step 4: Full Layer 0 Output ===")
# Get position embeddings for RoPE
seq_len = input_ids.shape[1]
position_ids = torch.arange(seq_len, device='cuda').unsqueeze(0)

with torch.no_grad():
    # HF full forward through layer 0
    hf_pos_emb = hf_model.model.rotary_emb(hf_embed, position_ids)
    hf_layer0_out = hf_model.model.layers[0](
        hf_embed,
        position_ids=position_ids,
        position_embeddings=hf_pos_emb,
    )

    # IC full forward through layer 0
    ic_rope = ic_model.model.rotary_pos_emb
    ic_layer0_out = ic_model.model.layers[0](
        ic_embed,
        attention_mask=None,
        rotary_pos_emb=ic_rope,
        position_ids=position_ids,
    )

layer0_diff = (hf_layer0_out.float() - ic_layer0_out.float()).abs()
print(f"Layer 0 output diff: max={layer0_diff.max().item():.6f}, mean={layer0_diff.mean().item():.6f}")

print("\n=== Done ===")
