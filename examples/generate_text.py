#!/usr/bin/env python3
"""Generate text from a trained model."""

from pathlib import Path

import torch
from ironcore.config import load_trainer_config
from ironcore import set_global_states, get_tokenizer
from ironcore.language_model import LanguageModel
from ironcore.checkpointing import load_checkpoint

def generate(model, tokenizer, prompt="Once upon a time", max_length=200, temperature=0.8, top_k=40):
    """Generate text from the model."""
    model.eval()

    # Determine device from model
    device = next(model.parameters()).device

    # Tokenize prompt
    input_ids = torch.tensor([tokenizer.encode(prompt)]).to(device)

    generated = input_ids.clone()

    with torch.no_grad():
        for _ in range(max_length):
            # Get logits
            output = model(generated, labels=None)  # Returns logits when labels=None
            logits = output[:, -1, :] / temperature

            # Top-k sampling
            if top_k > 0:
                indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
                logits[indices_to_remove] = float('-inf')

            # Sample
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            # Append
            generated = torch.cat([generated, next_token], dim=1)

            # Stop at EOS
            if next_token.item() == tokenizer.eos_token_id:
                break

    return tokenizer.decode(generated[0].tolist())

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default='models/example_fix_0/step_1000')
    parser.add_argument('--prompt', type=str, default='Once upon a time')
    parser.add_argument('--max-length', type=int, default=200)
    args = parser.parse_args()

    # Load config
    config = load_trainer_config()
    set_global_states(config)
    tokenizer = get_tokenizer()

    # Determine device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model
    model = LanguageModel(config).to(device)
    checkpoint_path = Path(args.checkpoint)
    config.trainer.model_path = str(checkpoint_path.parent)
    try:
        step_str = checkpoint_path.name.split('_')[-1]
        step = int(step_str)
    except (ValueError, IndexError) as e:
        raise ValueError(f"Could not extract step number from checkpoint path: {args.checkpoint}") from e

    step_loaded = load_checkpoint(config, model, optimizer=None, lr_scheduler=None, step=step)
    if step_loaded < 0:
        print(f"Error: Could not load checkpoint from {args.checkpoint}. Aborting.")
        return

    # Generate
    print(f"\nPrompt: {args.prompt}\n")
    print("="*80)
    text = generate(model, tokenizer, args.prompt, args.max_length)
    print(text)
    print("="*80)

if __name__ == '__main__':
    main()
