# PEFT 가이드: LoRA

IronCore의 LoRA는 베이스 모델을 고정한 채 어텐션과 MLP 레이어에 저랭크 어댑터 행렬(`lora_A`, `lora_B`)을 추가합니다. 어댑터는 샤딩 없이 TP 랭크 전체에 **복제**됩니다. 복제 방식은 저랭크 행렬 분할로 인한 근사 오류 없이 수학적 정확성을 유지하며, 체크포인팅과 그레이디언트 동기화에 TP-aware 처리가 따로 필요 없습니다.

## 설정

### 기본 설정

PEFT 설정 파일을 만듭니다 (예: `configs/peft/lora_default.yaml`):

```yaml
# LoRA 랭크 — 어댑터 크기 조절
r: 8

# 스케일링 팩터 — alpha/r이 업데이트 크기를 결정
alpha: 16.0

# LoRA 활성화에 적용할 드롭아웃 (0.0 = 드롭아웃 없음)
dropout: 0.0

# LoRA를 적용할 레이어
target_modules:
  - q_proj      # 어텐션 쿼리 프로젝션
  - v_proj      # 어텐션 값 프로젝션
  - o_proj      # 어텐션 출력 프로젝션
  - up_proj     # MLP up 프로젝션
  - down_proj   # MLP down 프로젝션
```

### 사용 가능한 target modules

어텐션: `q_proj`, `k_proj`, `v_proj` (k와 v는 레이어를 공유), `o_proj`.

MLP: `up_proj` (gate_proj는 동일하게 처리), `down_proj`.

### 훈련 설정

```yaml
model: llama-7b          # 모델 설정
data: alpaca_sft         # 데이터셋 설정
peft: lora_default       # PEFT 설정 참조

trainer:
  micro_batch_size: 8
  train_batch_size: 128
  tensor_model_parallel_size: 2
  sequence_chunk_size: 512  # 선택: 비동기 청킹
```

또는 인라인으로:

```yaml
peft:
  method: lora
  lora:
    r: 16
    alpha: 32
    dropout: 0.1
    target_modules: ["q_proj", "v_proj", "k_proj", "o_proj"]
```

## 사용 예시

### 기본 LoRA 훈련

```bash
torchrun --nproc_per_node=2 ironcore/cli/train.py --config-path configs/train_lora.yaml
```

예상 출력:
```
Freezing base model parameters for PEFT method: lora
Trainable parameters: 8,388,608 / 7,016,169,472 (0.12%)
```

### 랭크별 설정

낮은 랭크 (`r=4`): 최소 파라미터, 빠른 훈련:
```yaml
peft:
  method: lora
  lora:
    r: 4
    alpha: 8
    target_modules: ["q_proj", "v_proj"]
```

높은 랭크 (`r=64`): 더 높은 용량, 느린 훈련:
```yaml
peft:
  method: lora
  lora:
    r: 64
    alpha: 128
    target_modules: ["q_proj", "v_proj", "k_proj", "o_proj", "up_proj", "down_proj"]
```

### 청킹 실행

```yaml
trainer:
  tensor_model_parallel_size: 2
  sequence_chunk_size: 512

peft:
  method: lora
  lora:
    r: 8
    alpha: 16
```

## 구현 세부 사항

### 순전파

선형 레이어 `Y = XW`에 대해 LoRA가 추가하는 것:

```
Y_lora = X @ W + (X @ A @ B) * scaling
```

`A`는 `[in_features, r]` (Kaiming uniform 초기화), `B`는 `[r, out_features]` (zero 초기화), `scaling = alpha / r`.

### 텐서 병렬 동작

**Column parallel** (출력 차원 샤딩):
```python
base_output = base_layer(x)  # [batch, seq, out/tp_size]
lora_output = lora(x)  # [batch, seq, out] — 복제
lora_shard = lora_output[..., rank * size : (rank + 1) * size]
return base_output + lora_shard
```

**Row parallel** (입력 차원 샤딩):
```python
base_partial, handle = base_layer(x, async_communication=True)
lora_output = lora(x)  # 전체 입력, 복제
handle.wait()
return base_partial + bias + lora_output
```

## 랭크와 타겟 선택

| 태스크 | r | alpha | 모듈 |
|---|---|---|---|
| 지시 따르기 | 8–16 | 16–32 | q, v, o |
| 도메인 적응 | 16–32 | 32–64 | q, k, v, o, up, down |
| 태스크 특화 | 4–8 | 8–16 | q, v |

메모리 대략 추정 (7B 모델):
- LoRA만: 전체 파인튜닝 메모리의 ~15–20%
- LoRA + 청킹: ~10–15%
- LoRA + TP=2 + 청킹: 2x24GB GPU에서 13B 가능

LoRA는 일반적으로 전체 파인튜닝보다 2–5배 높은 학습률이 필요합니다:

```yaml
optim:
  max_lr: 5e-4   # 전체 파인튜닝의 ~1e-4와 비교
```

## 테스트

```bash
# TP 정확성
python tests/test_lora_tp_correctness.py --mode save_weights
torchrun --nproc_per_node=2 tests/test_lora_tp_correctness.py --mode load_and_compare

# 비동기 청킹
python tests/test_lora_async.py --tp 1
torchrun --nproc_per_node=2 tests/test_lora_async.py --tp 2

# 체크포인트 저장/로딩
python tests/test_lora_checkpoint.py --test save_load_tp1
torchrun --nproc_per_node=2 tests/test_lora_checkpoint.py --test universal_checkpoint
```

예상: TP=1과 TP=2 출력이 `atol=1e-1` 이내에서 일치; `lora_A`/`lora_B` 파라미터만 그레이디언트 수신; 훈련 가능한 파라미터가 전체의 5% 미만.

## 트러블슈팅

**대용량 배치에서 높은 메모리 사용.** `micro_batch_size`를 줄이거나 `sequence_chunk_size: 512`를 활성화하세요.

**손실이 감소하지 않는 경우.** LoRA는 전체 파인튜닝보다 높은 학습률이 필요합니다. `max_lr: 5e-4`로 시도하고 더 강한 업데이트를 위해 `alpha`를 높이세요 (예: 32).

**TP 랭크 간 출력 차이.** TP 정확성 테스트를 실행하세요. 일반적인 원인은 LoRA 가중치가 모든 랭크에 동일하게 로딩되지 않은 것입니다.

**체크포인트 로딩 실패.** 로딩 설정의 `r`, `alpha`, `target_modules`가 훈련 시 사용된 것과 정확히 일치해야 합니다.

## 성능 벤치마크

A100에서의 대략적인 속도 (7B 모델):

| 설정 | GPU당 메모리 | 토큰/초 |
|---|---|---|
| 전체 파인튜닝 (TP=1) | 56 GB | 1200 |
| LoRA (TP=1) | 15 GB | 1150 |
| LoRA (TP=2) | 8 GB | 2200 |
| LoRA + 청킹 (TP=2) | 6 GB | 2000 |

## 참고 문헌

- LoRA 논문: [Hu et al., 2021](https://arxiv.org/abs/2106.09685)
