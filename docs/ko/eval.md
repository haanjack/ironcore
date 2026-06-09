# 평가

## 내장 평가기

`trainer.do_eval: true`일 때 훈련 중 두 가지 평가 메커니즘이 실행됩니다:

1. **평가 손실 / 퍼플렉서티** — 훈련 데이터의 held-out 평가 분할에서 실행됩니다. `eval_loss`와 `eval_accuracy` (토큰별 정확도)로 기록됩니다.
2. **태스크 평가기** (예: HellaSwag) — `data.eval_datasets`에서 로드됩니다. 각 평가기는 훈련 데이터 분포와 독립적으로 실행됩니다.

## HellaSwag

상식 NLI 정확도를 측정합니다. 각 질문은 맥락과 네 개의 후보 연속을 제공하며, 모델은 토큰별 교차 엔트로피 손실이 가장 낮은 연속을 선택합니다.

활성화 방법:

```yaml
trainer:
  do_eval: true

data:
  eval_datasets:
    - name: hellaswag
      source: Rowan/hellaswag
      max_samples: 1000
```

## 커스텀 평가기 추가

1. `ironcore/eval/tasks/<name>.py` 생성 (소문자 파일 이름; 설정의 `name` 필드와 일치해야 함).
2. `ironcore/eval/tasks/base_task.py`의 `Task`를 서브클래싱.
3. 다음 메서드 구현:
   - `_preprocess(examples)` — HF `datasets.map()` 함수; 확장된 dict 반환.
   - `_get_batch(batch)` — 데이터로더 배치에서 입력과 레이블 추출.
   - `_do_predict(model, inputs)` — 모델 실행 후 샘플별 점수 반환.
   - `_get_score(...)` — 점수를 집계하고 최소한 `"score"` 키를 포함하는 메트릭 dict 반환.
4. 설정의 `data.eval_datasets`에 태스크 추가.

## 설정 레퍼런스

| 필드 | 기본값 | 설명 |
|---|---|---|
| `trainer.do_eval` | `false` | 훈련 중 평가 활성화 |
| `trainer.eval_batch_size` | `null` | 평가기 배치 크기 (`micro_batch_size`로 대체) |
| `operation.eval_interval` | `100` | N 스텝마다 평가 |
| `operation.eval_samples` | `100` | 평가 손실 평가에 사용할 샘플 수 |
| `data.eval_datasets` | `[]` | 태스크 평가기 설정 목록 (`name`, `source`, `max_samples`) |
