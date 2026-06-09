# 추론 및 생성

> 이 가이드는 생성 설정과 KV 캐시 옵션을 다룹니다. 캐시 데이터 구조, paged attention 메커니즘, TP-aware 레이아웃은 [KV 캐시 & 추론 시스템 설계](design/kv-cache.md)를 참고하세요.

## 생성

`LanguageModel.generate()`는 표준 prefill → decode 패턴을 따릅니다: 프롬프트를 하나의 순전파로 처리한 후, 캐싱된 KV 텐서를 재사용해 토큰을 하나씩 생성합니다.

### 샘플링 파라미터

```yaml
alignment:
  generation:
    max_new_tokens: 512
    temperature: 1.0       # 1.0 = 스케일링 없음; 낮을수록 더 선명
    top_p: 0.9             # nucleus 샘플링; 1.0 = 비활성화
    top_k: 0               # top-k 컷오프; 0 = 비활성화
    do_sample: true        # false = greedy (argmax)
```

이 설정은 GRPO 롤아웃과 직접 `generate()` 호출 모두에서 사용됩니다.

## KV 캐시 옵션

IronCore는 서로 다른 워크로드를 위해 두 가지 KV 캐시 구현을 제공합니다:

| 캐시 | 사용 사례 | 접두사 공유 |
|---|---|---|
| `KVCacheManager` | 표준 추론, 평가 | 없음 |
| `BlockKVCacheManager` | GRPO 롤아웃 생성 | 있음 — 그룹당 프롬프트 KV 사본 하나 |

### 표준 KV 캐시 (`KVCacheManager`)

레이어당 밀집 `[batch, max_seq_len, kv_groups, head_dim]` 버퍼를 사전 할당합니다. `trainer.use_kv_cache_in_eval: true`일 때 평가 중 사용됩니다.

```yaml
model:
  kv_cache:
    max_seq_length: 2048   # 최대 캐시 시퀀스 길이
```

### GRPO용 Paged KV 캐시 (`BlockKVCacheManager`)

고정 크기 페이지에 참조 카운팅을 사용해 프롬프트 KV를 저장합니다. 프롬프트의 G개 완성이 프롬프트의 KV 사본 하나를 공유합니다. G개의 동일한 사본을 저장하는 낭비를 피할 수 있습니다. GRPO 롤아웃 경로의 `generate_rollouts_paged()`를 사용할 때 자동으로 활성화됩니다.

페이지 크기와 블록 테이블 차원은 `model.kv_cache.max_seq_length`에서 파생됩니다.

## FlashAttention

FlashAttention은 기본적으로 활성화되어 있으며, 표준 시퀀스와 bin-packed SFT 시퀀스 모두를 처리합니다 (샘플별 position ID를 통한 블록 대각 어텐션).

```yaml
trainer:
  use_flash_attn: true   # 기본값
```

CUDA 지원 장치 없이 실행하거나 디버깅할 때만 비활성화하세요.

## 설정 레퍼런스

| 필드 | 기본값 | 설명 |
|---|---|---|
| `trainer.use_flash_attn` | `true` | FlashAttention 커널 사용 |
| `trainer.use_kv_cache_in_eval` | `true` | 평가 중 `KVCacheManager` 사용 |
| `model.kv_cache.max_seq_length` | `2048` | 최대 캐시 시퀀스 길이; 블록 테이블 행 크기도 설정 |
| `alignment.generation.max_new_tokens` | `512` | 최대 생성 토큰 수 |
| `alignment.generation.temperature` | `1.0` | 샘플링 온도 |
| `alignment.generation.top_p` | `0.9` | Nucleus 샘플링 임계값 |
| `alignment.generation.top_k` | `0` | Top-k 컷오프 (0 = 비활성화) |
| `alignment.generation.do_sample` | `true` | 확률적 vs greedy 디코딩 |
