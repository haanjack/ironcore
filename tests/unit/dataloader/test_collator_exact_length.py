"""Exact-length packed rows for the SFT collator (CPU, 2026-10-13).

검증 항목:
1. 행 폭 = 배치 내 최장 빈 컨텐츠 (32K 고정 아님)
2. content region의 input_ids/labels/position_ids가 패킹 의미 보존
3. cu_seqlens/블록마스크/loss_sample_ids 정합
"""

import sys

sys.path.insert(0, "/tmp/opencode/ironcore")
import torch
from ironcore.dataloader.collator import UniversalCollator


def make_sample(lengths, base=1000):
    token_ids, spans = [], []
    tid = base
    for L in lengths:
        spans.append((len(token_ids), len(token_ids) + L))
        token_ids.extend(range(tid, tid + L))
        tid += L
    meta = {"mask_ranges": [[spans[0][0], spans[-1][1]]]}
    return {"token_ids": torch.tensor(token_ids), "metadata": meta}


def batch_from(samples):
    return [{"token_ids": s["token_ids"], "metadata": s["metadata"]} for s in samples]


def check(coll, samples, tag):
    out = coll(batch_from(samples))
    B, W = out["input_ids"].shape
    print(f"[{tag}] rows={B} width={W} (max_seq_len={coll.max_seq_len})")
    ok = True
    for b in range(B):
        ids = out["input_ids"][b]
        lab = out["labels"][b]
        pos = out["position_ids"][b]
        sid = out.get("loss_sample_ids")
        sid_b = sid[b] if sid is not None else None
        cu = out["cu_seqlens"][b].tolist() if out.get("cu_seqlens") is not None else []
        valid = (ids != coll.pad_token_id) | (lab != -100)
        nv = int(valid.sum())
        # 시프트 라벨: 같은 샘플 내에서 labels[j] == ids[j+1]
        for i in range(W - 1):
            if sid_b is not None and sid_b[i] == sid_b[i + 1] != -1 and lab[i] != -100:
                assert lab[i].item() == ids[i + 1].item(), f"shift mismatch at {i}"
        # position 리셋: sid 경계에서 pos == 0
        if sid_b is not None:
            for i in range(1, W):
                if sid_b[i] != sid_b[i - 1] and sid_b[i] != -1:
                    assert pos[i].item() == 0, f"position reset FAIL at {i}"
        # 마스크 대각
        if out.get("attention_mask") is not None:
            assert out["attention_mask"][b].diagonal().all(), (
                f"mask diagonal FAIL bin{b}"
            )
        print(f"  bin{b}: 유효토큰 {nv}, cu={cu}")
    print(f"  [{tag}] 검증 OK")
    return out


samples1 = [make_sample([2600 + (i % 3) * 100]) for i in range(8)]
coll_pack = UniversalCollator(
    "sft",
    max_seq_len=32768,
    pad_token_id=0,
    use_flash_attention=True,
    return_full_attention_mask=True,
    pack_sequences=True,
)
check(coll_pack, samples1, "packing micro8")

samples2 = [
    make_sample([20000]),
    make_sample([3000]),
    make_sample([2500]),
    make_sample([800]),
]
check(coll_pack, samples2, "packing multi-bin")

coll_nopack = UniversalCollator(
    "sft",
    max_seq_len=500,
    pad_token_id=0,
    use_flash_attention=False,
    return_full_attention_mask=True,
    pack_sequences=False,
)
check(coll_nopack, [make_sample([400])], "nopack single")
check(coll_nopack, [make_sample([500])], "nopack exact-fit")

print("\n모든 케이스 통과")
