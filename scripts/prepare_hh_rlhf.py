import json
from pathlib import Path

from datasets import load_dataset
from tqdm import tqdm


def parse_dialogue(text):
    """
    Parse Anthropic HH-RLHF "Human: ... Assistant: ..." format into chat list.
    """
    # Split by the standardized delimiters
    parts = text.split("\n\n")

    messages = []

    for part in parts:
        part = part.strip()
        if not part:
            continue

        if part.startswith("Human:"):
            content = part[len("Human:") :].strip()
            messages.append({"role": "user", "content": content})
        elif part.startswith("Assistant:"):
            content = part[len("Assistant:") :].strip()
            messages.append({"role": "assistant", "content": content})

    return messages


def prepare_dataset():
    print("Loading Anthropic HH-RLHF dataset...")
    dataset = load_dataset("anthropic/hh-rlhf")

    output_dir = Path("data/local/hh_rlhf")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Process Splits
    for split in ["train", "test"]:
        print(f"Processing {split} split...")
        data = dataset[split]

        # We need two versions:
        # 1. SFT: treating "chosen" as the target conversation
        # 2. DPO: keeping both "chosen" and "rejected"

        sft_rows = []
        dpo_rows = []

        for item in tqdm(data):
            chosen_text = item["chosen"]
            rejected_text = item["rejected"]

            try:
                chosen_msgs = parse_dialogue(chosen_text)
                rejected_msgs = parse_dialogue(rejected_text)

                # SFT Data
                sft_rows.append({"messages": chosen_msgs})

                # DPO Data
                dpo_rows.append({"chosen": chosen_msgs, "rejected": rejected_msgs})
            except Exception:
                continue

        # Save SFT
        sft_path = output_dir / f"sft_{split}.jsonl"
        with open(sft_path, "w") as f:
            for row in sft_rows:
                f.write(json.dumps(row) + "\n")
        print(f"Saved SFT {split} to {sft_path}")

        # Save DPO
        dpo_path = output_dir / f"dpo_{split}.jsonl"
        with open(dpo_path, "w") as f:
            for row in dpo_rows:
                f.write(json.dumps(row) + "\n")
        print(f"Saved DPO {split} to {dpo_path}")


if __name__ == "__main__":
    prepare_dataset()
