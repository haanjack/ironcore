#!/usr/bin/env python
"""Debug script to compare HF and IronCore generation."""

import sys
import os

# Set up paths
script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(script_dir)
sys.path.insert(0, project_dir)
os.chdir(project_dir)

os.environ['WORLD_SIZE'] = '2'
sys.argv = ['debug', '--config-path', 'configs/grpo_gsm8k_smoke.yaml']

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer as HFAutoTokenizer

# HF test
print("=== HF Model ===")
hf_model = AutoModelForCausalLM.from_pretrained(
    'Qwen/Qwen2.5-0.5B-Instruct', torch_dtype=torch.bfloat16, device_map='auto'
)
hf_tokenizer = HFAutoTokenizer.from_pretrained('Qwen/Qwen2.5-0.5B-Instruct')

messages = [{'role': 'user', 'content': 'What is 2+2?'}]
hf_enc = hf_tokenizer.apply_chat_template(
    messages, add_generation_prompt=True, return_tensors='pt'
)
# BatchEncoding is dict-like but not a dict - check for 'input_ids' key
if hasattr(hf_enc, 'input_ids'):
    hf_input = hf_enc['input_ids']
elif isinstance(hf_enc, dict):
    hf_input = hf_enc['input_ids']
else:
    hf_input = hf_enc
hf_input = hf_input.to(hf_model.device)

with torch.no_grad():
    hf_logits = hf_model(hf_input).logits
    hf_top5 = hf_logits[0, -1, :].topk(5)
    print('HF top-5 tokens at last position:')
    for i in range(5):
        tok = hf_tokenizer.decode([hf_top5.indices[i].item()])
        print(f"  {tok!r} (id={hf_top5.indices[i].item()}, logit={hf_top5.values[i].item():.2f})")

# IronCore test
print("\n=== IronCore Model ===")
from ironcore import get_tokenizer
from ironcore.config import load_trainer_config
from ironcore.global_vars import set_global_states
from ironcore.language_model import LanguageModel
from ironcore.parallel.parallel_states import initialize_model_parallel
from ironcore.checkpointing.hf_interop import load_from_huggingface
from huggingface_hub import snapshot_download

config = load_trainer_config()
initialize_model_parallel(1, timeout_in_minutes=10)
set_global_states(config)
ic_model = LanguageModel(config).to('cuda')
cache_dir = snapshot_download(config.trainer.load_from_hf)
load_from_huggingface(cache_dir, ic_model, 'qwen2', strict=False)
ic_model.eval()

ic_tokenizer = get_tokenizer()
messages_ic = [{'role': 'user', 'content': 'What is 2+2?'}]
ic_enc = ic_tokenizer.apply_chat_template(
    messages_ic, add_generation_prompt=True, return_tensors='pt'
)
# BatchEncoding is dict-like but not a dict - check for 'input_ids' key
if hasattr(ic_enc, 'input_ids'):
    ic_input = ic_enc['input_ids']
elif isinstance(ic_enc, dict):
    ic_input = ic_enc['input_ids']
else:
    ic_input = ic_enc
ic_input = ic_input.to(ic_model.device)

print(f"IC input IDs: {ic_input}")

with torch.no_grad():
    ic_logits = ic_model(ic_input)
    ic_top5 = ic_logits[0, -1, :].topk(5)
    print('IC top-5 tokens at last position:')
    for i in range(5):
        tok = ic_tokenizer.decode([ic_top5.indices[i].item()])
        print(f"  {tok!r} (id={ic_top5.indices[i].item()}, logit={ic_top5.values[i].item():.2f})")

# Compare embedding weights
print("\n=== Embedding Comparison ===")
hf_embed = hf_model.model.embed_tokens.weight
ic_embed = ic_model.embedding.word_embeddings.weight
print(f"HF embed shape: {hf_embed.shape}, dtype: {hf_embed.dtype}")
print(f"IC embed shape: {ic_embed.shape}, dtype: {ic_embed.dtype}")

# Check if values match for first few tokens
print(f"HF embed[0, :5]: {hf_embed[0, :5]}")
print(f"IC embed[0, :5]: {ic_embed[0, :5]}")

# Check if they're close
diff = (hf_embed.float() - ic_embed.float()).abs()
print(f"Embedding diff max: {diff.max().item():.6f}, mean: {diff.mean().item():.6f}")
