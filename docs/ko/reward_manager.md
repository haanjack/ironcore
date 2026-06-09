# RewardManager & YAML 규칙 템플릿

## 아키텍처

```
AlignmentConfig
  ├── reward: RewardConfig          # 레거시 (deprecated, 여전히 작동)
  └── reward_manager: RewardManagerConfig   # 신규 (우선 적용)
        └── functions: list[RewardFunctionEntry]
              ├── type: "rule_template"  → TemplateRuleReward.from_yaml(path)
              ├── type: "reward_model"   → RewardModelFunction(backend=...)
              └── type: "math"|"code"|…  → get_reward_function() (레거시 클래스)
```

### 컴포넌트 역할

| 컴포넌트 | 파일 | 역할 |
|---|---|---|
| `RewardManager` | `ironcore/alignment/rewards/manager.py` | 가중치 레지스트리/오케스트레이터. `RewardFunction`을 상속해 `RewardWorkerPool`이 변경 없이 사용 가능. |
| `TemplateRuleReward` | `ironcore/alignment/rewards/template.py` | YAML 기반 규칙 보상. `MathRewardFunction`, `FormatRewardFunction`, `StrictFormatRewardFunction`을 설정 기반 로직으로 대체. |
| `RewardModelFunction` | `ironcore/alignment/rewards/model.py` | 분류기 헤드 RM 채점 (스칼라 출력). 백엔드: `local_endpoint`, `api`, `local_inference`. |
| `RewardFunctionEntry` | `ironcore/config/config_alignment.py` | 단일 보상 함수 항목의 설정 데이터클래스. |
| `RewardManagerConfig` | `ironcore/config/config_alignment.py` | 항목 목록 + 워커 설정을 담는 설정 데이터클래스. |

### 연결 방식

```
GRPOTrainer._post_checkpoint_load()
  │
  ├─ reward_manager 설정 있음?
  │   YES → RewardManager.from_config(reward_manager_cfg)
  │   NO  → RewardManager.from_legacy_config(reward_cfg)  # 하위 호환
  │
  └─ RewardWorkerPool(reward_fn=manager, ...)
       └─ manager.compute() 호출 (prompt, completion, metadata)당
            └─ 모든 등록된 함수의 가중 합
```

`RewardManager`는 `RewardFunction`을 상속하므로 나머지 훈련 루프는 변경되지 않습니다.

---

## 설정 레퍼런스

### RewardManagerConfig

```yaml
alignment:
  method: grpo
  reward_manager:
    num_workers: 4        # 병렬 보상 계산을 위한 스레드 풀 크기
    timeout: 30           # 기본 보상(0.5) 반환 전 대기 시간(초)
    functions:            # RewardFunctionEntry 목록
      - name: correctness
        type: rule_template
        weight: 0.6
        rule_template: configs/rewards/math_gsm8k.yaml
      - name: format
        type: rule_template
        weight: 0.4
        rule_template: configs/rewards/format_deepseek.yaml
```

### RewardFunctionEntry 필드

| 필드 | 타입 | 기본값 | 설명 |
|---|---|---|---|
| `name` | str | `"default"` | 사람이 읽을 수 있는 이름 (로깅에 사용) |
| `type` | str | `"rule_template"` | `"rule_template"` \| `"reward_model"` \| `"math"` \| `"code"` \| `"api"` \| `"format"` \| ... |
| `weight` | float | `1.0` | 가중 합에서의 가중치 |
| `rule_template` | str \| None | `None` | YAML 규칙 파일 경로 (`type="rule_template"`일 때) |
| `rm_backend` | str | `"local_endpoint"` | 보상 모델 백엔드 (`type="reward_model"`일 때): `"local_endpoint"` \| `"api"` \| `"local_inference"` |
| `api_provider` | str | `"openai"` | `type="api"` 또는 `rm_backend="api"`의 API 제공자 |
| `api_model` | str \| None | `None` | API 호출에 사용할 모델 이름 |
| `local_endpoint` | str | `"http://localhost:8000/v1"` | 로컬 엔드포인트 URL |
| `local_model_path` | str \| None | `None` | 로컬 추론용 HF 모델 경로 |
| `local_device` | str | `"cuda:0"` | 로컬 추론용 장치 |
| `local_dtype` | str | `"bfloat16"` | 로컬 추론용 dtype |
| `keyword` | str | `""` | `type="keyword"` 또는 `type="soft_keyword"`용 키워드 |

---

## YAML 규칙 템플릿

규칙 템플릿은 `configs/rewards/`에 위치하며 `TemplateRuleReward.from_yaml()`로 로딩됩니다.

### 모드: `answer_match`

정규식 패턴으로 완성에서 답을 추출하고, 정규화 후 정답과 비교합니다.

```yaml
# configs/rewards/math_gsm8k.yaml
mode: answer_match
answer_patterns:
  - '####\s*(.+)'
  - '\\boxed\{(.+?)\}'
  - '[Aa]nswer:\s*(.+)'
  - '[Tt]he answer is\s*(.+)'
normalization:
  lowercase: true
  strip_chars: ", _$"
  strip_trailing_period: true
fallback_last_number: true     # 패턴이 없으면 텍스트의 마지막 숫자 사용
scoring:
  correct: 1.0                 # 정규화 후 정확히 일치
  partial: 0.1                 # 추출했지만 틀림
  no_answer: 0.0               # 답을 추출하지 못함
```

### 모드: `tag_check`

필수 태그의 존재를 확인합니다. 구조화된 출력 강제에 유용합니다.

```yaml
# configs/rewards/format_cot.yaml
mode: tag_check
required_tags:
  - "<thought>"
  - "</thought>"
  - "<answer>"
  - "</answer>"
scoring:
  all_present: 0.0             # 모든 태그 발견 시 점수
  per_missing_tag: -0.1        # 누락 태그당 패널티
```

### 모드: `regex_match`

전체 정규식 일치 기반 이진 보상. 엄격한 형식 강제에 유용합니다.

```yaml
# configs/rewards/format_deepseek.yaml
mode: regex_match
pattern: '<think>.*?</think>\s*####\s*.*'
pattern_flags:
  - DOTALL                     # .이 개행 문자와 일치
scoring:
  match: 1.0
  no_match: 0.0
```

---

## 사용 예시

### 예시 1: 형식 강제가 있는 GSM8K (일반적인 설정)

```yaml
alignment:
  method: grpo
  grpo_group_size: 8
  reward_manager:
    num_workers: 4
    timeout: 30
    functions:
      - name: correctness
        type: rule_template
        weight: 0.6
        rule_template: configs/rewards/math_gsm8k.yaml
      - name: format
        type: rule_template
        weight: 0.4
        rule_template: configs/rewards/format_deepseek.yaml
```

### 예시 2: 보상 모델 + 규칙 기반 형식 확인

```yaml
alignment:
  method: grpo
  reward_manager:
    functions:
      - name: helpfulness_rm
        type: reward_model
        weight: 0.5
        rm_backend: local_endpoint
        local_endpoint: http://localhost:8000/v1
      - name: format
        type: rule_template
        weight: 0.3
        rule_template: configs/rewards/format_cot.yaml
      - name: correctness
        type: rule_template
        weight: 0.2
        rule_template: configs/rewards/math_gsm8k.yaml
```

### 예시 3: 레거시 + 신규 타입 혼합

```yaml
alignment:
  method: grpo
  reward_manager:
    functions:
      - name: code_exec
        type: code
        weight: 0.7
      - name: format
        type: rule_template
        weight: 0.3
        rule_template: configs/rewards/format_cot.yaml
```

### 예시 4: 커스텀 템플릿 (복사 후 수정)

```bash
cp configs/rewards/math_gsm8k.yaml configs/rewards/math_custom.yaml
# math_custom.yaml 편집: 도메인별 패턴 추가, 채점 변경
```

```yaml
alignment:
  method: grpo
  reward_manager:
    functions:
      - name: correctness
        type: rule_template
        weight: 1.0
        rule_template: configs/rewards/math_custom.yaml
```

### 예시 5: 레거시 설정 (변경 없이 그대로 작동)

```yaml
alignment:
  method: grpo
  reward:
    type: composite_math
    num_workers: 4
    timeout: 30
```

---

## 커스텀 규칙 템플릿 만들기

1. 모드 선택: `answer_match`, `tag_check`, 또는 `regex_match`
2. `configs/rewards/`에 YAML 파일 생성
3. 훈련 설정에서 참조

커스텀 답 추출 예시:

```yaml
# configs/rewards/science_answer.yaml
mode: answer_match
answer_patterns:
  - 'Final answer:\s*(.+)'
  - 'Result:\s*(.+)'
  - '=\s*(.+?)(?:\s|$)'
normalization:
  lowercase: true
  strip_chars: " "
  strip_trailing_period: true
fallback_last_number: false
scoring:
  correct: 1.0
  partial: 0.0
  no_answer: 0.0
```

---

## RewardModelFunction 백엔드

`RewardModelFunction` (`type: reward_model`)은 모델에서 보상 점수를 얻기 위한 세 가지 백엔드를 지원합니다.

| 백엔드 | 동작 | 요구 사항 |
|---|---|---|
| `local_endpoint` | `{endpoint}/reward`에 POST, `{"reward": float}` 또는 `{"score": float}` 기대 | 실행 중인 vLLM/SGLang 서버 |
| `api` | 프롬프트+완성을 채팅 메시지로 전송, 응답에서 스칼라 파싱 | `OPENAI_API_KEY` 환경 변수 |
| `local_inference` | GPU에 `AutoModelForSequenceClassification` 로드, 보상 헤드 출력 읽기 | HF 모델 가중치 + GPU |

### 예시: 외부 API 보상 모델

```yaml
alignment:
  method: grpo
  reward_manager:
    functions:
      - name: helpfulness
        type: reward_model
        rm_backend: api
        api_provider: openai        # openai | anthropic | google | zhipu
        api_model: gpt-4o-mini
        weight: 1.0
```

`OPENAI_API_KEY` 환경 변수가 필요합니다.

### 예시: 로컬 서버 (vLLM/SGLang)

```yaml
alignment:
  method: grpo
  reward_manager:
    functions:
      - name: helpfulness
        type: reward_model
        rm_backend: local_endpoint
        local_endpoint: http://localhost:8000/v1
        weight: 1.0
```

### 예시: 로컬 GPU 추론 (서버 없음, API 키 없음)

```yaml
alignment:
  method: grpo
  reward_manager:
    functions:
      - name: helpfulness
        type: reward_model
        rm_backend: local_inference
        local_model_path: OpenAssistant/reward-model-deberta-v3-large-v2
        local_device: cuda:0
        local_dtype: bfloat16
        weight: 1.0
```

HuggingFace `AutoModelForSequenceClassification` 모델을 GPU에 직접 로드합니다. 서버나 API 키가 필요 없습니다. 모델은 처음 한 번만 다운로드되어 `~/.cache/huggingface/`에 캐싱됩니다.
