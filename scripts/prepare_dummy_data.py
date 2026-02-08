
from pathlib import Path

import numpy as np


def create_dummy_data():
    # Adjusted output path to match get_dataset_output_path logic
    output_dir = Path("data/preprocessed/openwebtext/pretrain")
    output_dir.mkdir(parents=True, exist_ok=True)

    bin_path = output_dir / "data.bin"
    idx_path = output_dir / "data.idx.npy"

    if bin_path.exists() and idx_path.exists():
        print("Dummy data already exists.")
        return

    print("Generating dummy data...")

    # 1M tokens of random data
    num_tokens = 1_000_000
    vocab_size = 50257 # GPT2

    # Generate random tokens
    tokens = np.random.randint(0, vocab_size, size=num_tokens, dtype=np.uint16)

    # Save .bin
    with open(bin_path, "wb") as f:
        tokens.tofile(f)

    # Create metadata
    # Let's pretend it's 1000 documents of 1000 tokens each
    num_docs = 1000
    doc_len = 1000

    metadata_dtype = np.dtype([
        ('offset', np.uint64),
        ('length', np.uint32),
        ('type', 'U20'),
        ('group_id', np.int64),
        ('mask_ranges', 'U500'),
    ])

    metadata_list = []
    current_offset = 0

    for _ in range(num_docs):
        metadata_list.append((
            current_offset,
            doc_len,
            'pretrain',
            -1,
            '[]'
        ))
        current_offset += doc_len

    metadata = np.array(metadata_list, dtype=metadata_dtype)
    np.save(idx_path, metadata)

    print(f"Dummy data created at {output_dir}")

if __name__ == "__main__":
    create_dummy_data()
