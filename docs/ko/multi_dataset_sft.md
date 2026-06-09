# 멀티 데이터셋 SFT 훈련

IronCore는 가중치 혼합을 사용한 여러 SFT 데이터셋 훈련을 지원합니다.

## 설정

데이터 설정 YAML에 여러 데이터셋을 정의합니다:

```yaml
data:
  task_type: sft
  datasets:
    - source: openai/gsm8k
      task_type: sft
      ratio: 0.3
      split: train

    - source: tatsu-lab/alpaca
      task_type: sft
      ratio: 0.5
      split: train

    - source: my-org/code-dataset
      task_type: sft
      ratio: 0.2
      split: train
```

## 가중치 혼합 동작 방식

믹서는 `dataset_size × ratio`에 비례해 샘플링합니다. `ratio`가 높을수록 데이터셋의 실질적 기여가 높아지지만, 실제 비율은 `ratio`와 데이터셋 크기 **모두**에 의존합니다:

```
실질적 기여 = (dataset_size × ratio) / sum(모든 dataset_size × ratio)
```

**예시:**
```
Dataset A: 1000 samples, ratio=0.2  →  weighted = 1000 × 0.2 = 200
Dataset B:  500 samples, ratio=0.6  →  weighted =  500 × 0.6 = 300
Dataset C:  200 samples, ratio=0.2  →  weighted =  200 × 0.2 =  40

실제 비율: A=37%, B=56%, C=7%
```

## 모범 사례

### 크기가 다른 데이터셋 균형 맞추기

크기에 관계없이 동등한 표현을 원하면 역크기 가중치를 사용합니다:

```python
# 크기 10000과 2000 데이터셋을 50/50으로:
ratio_a = 1 / 10000  # 0.0001
ratio_b = 1 / 2000   # 0.0005
# 정규화: ratio_a=0.17, ratio_b=0.83
```

### 특정 데이터셋 강조

```yaml
data:
  datasets:
    - source: math_reasoning
      ratio: 2.0    # 더 높은 실질적 기여
    - source: general_chat
      ratio: 0.5
```

### 대규모 데이터셋 서브샘플링

```yaml
data:
  datasets:
    - source: huge_dataset
      max_samples: 10000   # 최대 10K 샘플로 제한
      ratio: 1.0
    - source: small_dataset
      ratio: 1.0
```

## 요약

- `data.datasets`를 통해 여러 SFT 데이터셋을 완전히 지원
- `ratio`는 데이터셋 크기에 상대적인 샘플링 가중치를 제어
- 실제 비율은 `ratio`와 데이터셋 크기 모두에 의존 — 대규모 데이터셋은 `max_samples`로 제한 가능
