# GRPO와 TRL의 학습 결과 비교 기록

이 폴더에는 IronCore와 Hugging Face TRL 0.29.0에 같은 초기 모델·completion·reward를 넣고, 실제 학습 루프가 만든 loss·gradient·가중치를 비교한 결과를 보관한다. 실험 설정과 수치 해석은 [16번 노트](../../16-grpo-trl-reference.md)에서 먼저 읽을 수 있다.

핵심 결과는 **135M 모델의 FP32 비교는 통과했고, BF16에서 여러 update를 거친 뒤의 gradient/loss 비교는 허용 오차를 벗어났다**는 것이다. 성공한 실험과 실패한 실험이 모두 들어 있다. Completion은 실험에서 지정했으므로 이 결과를 실제 생성 정답률의 개선으로 해석하지 않는다.

## 어떤 폴더를 읽으면 되는가

폴더 이름의 `v1`, `final2`, `diag`는 실행 당시 사용한 이름이다. 아래 표에서 각 실행의 목적과 자료를 구분한다. Update 수는 optimizer가 가중치를 갱신한 횟수다.

| 폴더 | 실험 | 결과 | 저장한 파일 |
|---|---|---|---|
| `cpu-final2` | 작은 76K 모델, CPU, FP32, 24 updates | 비교 통과 | rank0.json, run.log |
| `dp2-v1` | 같은 작은 모델, 2-GPU, FP32, 24 updates | 비교 통과 | rank0.json, rank1.json, run.log |
| `135m-dp2-fp32-v1` | 실제 135M 모델, 2-GPU, FP32, 8 updates | 비교 통과 | rank0.json, rank1.json, run.log |
| `135m-dp2-bf16-diag` | 실제 135M 모델, 2-GPU, BF16, 8 updates; 모든 update의 차이를 기록 | gradient/loss 비교 실패 | rank0.json, rank1.json, run.log |
| `135m-hf-dp2-bf16-v1` | 두 trainer에 같은 HF 모델 구현을 넣어 비교, 2-GPU, BF16, 8 updates | 여러 update 뒤 gradient/loss 비교 실패 | rank0.json, rank1.json, run.log |
| `hf-cpu-v3` | 위 HF 모델 연결 방식의 보조 검사, 작은 모델, CPU, 8 updates | 비교 통과 | rank0.json, run.log |
| `135m-dp2-bf16-v1` | BF16 비교의 첫 실행; 상세 기록 기능을 추가하기 전 | 학습은 완료했지만 첫 불일치에서 결과 비교 중단 | **run.log만 있음** |

`135m-dp2-bf16-v1/run.log`도 저장소에 포함한다. 처음 BF16 비교가 어디서 중단됐고 왜 상세 기록을 남기는 후속 실행을 했는지 추적하기 위한 기록이다. BF16의 전체 수치 결과를 보려면 이 초기 로그보다 **`135m-dp2-bf16-diag/rank0.json`**을 읽으면 된다.

## JSON 결과 파일 읽기

`rank0.json`과 `rank1.json`은 각각 GPU/process rank 0과 1의 결과 파일이다. CPU 단일 process 실험에는 rank0.json만 있다. 성공과 실패에 같은 파일 형식을 사용한다.

| 필드 | 뜻 |
|---|---|
| `status` | `passed`: 설정한 비교를 모두 통과. `failed`: 하나 이상의 비교가 허용 오차를 벗어남 |
| `errors` | Update별 loss 차이와 gradient·가중치의 최대 절대 차이 |
| `failures` | 실패한 update 번호, 비교 대상, parameter 이름, 실제 차이와 허용 오차. 통과한 결과에서는 비어 있으며, 초기 형식의 결과에는 이 필드가 없음 |
| `losses` | 두 trainer의 update별 loss 값 |
| `candidate_scores` | 지정한 두 completion의 학습 전후 확률. 전체 생성 성공률이 아님 |
| `initial_weights_exact` | 두 구현이 같은 초기 trainable weights에서 시작했는지 확인한 결과 |
| `reference_weights_exact_frozen` | Reference weights가 학습 중 바뀌지 않았는지 확인한 결과 |
| `command`, `experiment_script_sha256`, `trl_source_sha256` | 실행한 스크립트와 기준 TRL 코드 버전을 추적하는 정보 |

예를 들어 `status: "failed"`와 함께 `failures`에 `step: 2`, `kind: "gradient"`, `parameter: "model.embed_tokens.weight"`가 있으면, 두 번째 update의 해당 gradient 비교가 실패했다는 뜻이다. JSON이 깨졌거나 저장에 실패했다는 뜻이 아니다. 초기 실행은 이런 결과 파일을 쓰기 전에 비교가 중단되어 run.log만 남았다.

`run.log`는 실행 중 출력한 학습 메시지와 오류 내용을 담는다. 비교 실패 시 스크립트는 결과 JSON을 기록한 뒤 종료 코드 1로 끝난다. 이는 자동 실행 도구에서도 비교 실패를 감지하도록 한 동작이다.

## 요약·그림·재현 자료

- `summary.json`: 위 실험들의 판정과 최대 오차를 모은 요약.
- `fp32_loss_trajectories.png` / `.svg`: FP32에서 두 trainer의 loss가 어떻게 변했는지 보여주는 그림.
- `precision_discrepancy.png` / `.svg`: FP32/BF16 비교의 update별 gradient 차이.
- `runtime.json`: 사용한 라이브러리 버전, GPU 범위, 실행 환경 설정.
- `source/initial`, `source/final`: 실험 도구의 두 버전. 각 폴더의 `source.tar.gz`는 코드 묶음, `source_hashes.json`은 코드 파일별 SHA256이다. `initial`은 최초 비교 코드, `final`은 실패한 비교도 끝까지 기록하고 같은 HF 모델로 진단하는 기능을 추가한 코드다. 각 결과의 script hash와 대조해 읽는다.
- `manifest.json`: 이 폴더에 보관한 파일의 SHA256과 크기. 파일이 나중에 바뀌었는지 확인하는 용도다.

작은 모델은 스크립트의 seed=83과 Llama 설정으로 재구성한다. 135M 모델은 [모델 revision·파일 hash 기록](../2026-10-07-remediation/huggingface_manifest.json)에 지정한 SmolLM2 checkpoint를 사용한다. 큰 모델 weights와 update별 전체 optimizer 상태는 이 폴더에 보관하지 않는다. 다시 실행하는 명령은 [16번 노트의 재현 방법](../../16-grpo-trl-reference.md#재현과-블로그에서의-결론)에 있다.
