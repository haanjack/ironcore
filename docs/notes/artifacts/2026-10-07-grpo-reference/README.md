# GRPO와 TRL의 학습 결과 비교 기록

실험 시작일: 2026-10-07. 최종 보완 판정일: 2026-10-08.

이 폴더에는 IronCore와 Hugging Face TRL 0.29.0에 같은 초기 모델·completion·reward를 넣고, 실제 학습 루프가 만든 loss·gradient·가중치를 비교한 결과를 보관한다. 실험 설정과 수치 해석은 [16번 노트](../../16-grpo-trl-reference.md)에서 먼저 읽을 수 있다.

**최종 판정은 코드 보완 완료이며, 이 기록에서 확인한 필수 수정 작업은 남아 있지 않다.** 같은 log-probability의 중복 계산으로 생기던 BF16 gradient 반올림을 수정한 뒤, 135M Native/FP32 비교와 같은 HF backbone/BF16 비교가 모두 통과했다. Native의 MoE/BF16 DP·TP·정확한 재개도 다시 통과했다.

서로 다른 Native/HF backbone의 BF16 trajectory는 일치하지 않는다. 같은 backbone에서는 보완 후 trainer 비교를 통과했고, 다른 backbone의 비교는 학습 전 출력부터 달랐다. 따라서 후자는 이번 GRPO trainer에서 고쳐야 할 미결 버그로 남기지 않고 **현재 제공하지 않는 backbone 간 수치 동등성 보장**으로 분류했다. 그 보장이 필요한 실험은 Native/FP32 설정을 사용한다. 상세 근거와 사용 조건은 [16번 노트의 최종 판정](../../16-grpo-trl-reference.md#최종-판정-보완-완료-문서에-명시한-조건에서-사용-가능)에 있다.

아래 초기 실패 자료는 수정과 판정의 근거를 추적하기 위한 기록이다. 현재 상태를 판단할 때는 **`postfix/`의 보완 후 결과**를 먼저 읽는다. Completion은 실험에서 지정했으므로 이 결과를 실제 생성 정답률의 개선으로 해석하지 않는다.

## 어떤 폴더를 읽으면 되는가

폴더 이름의 `v1`, `final2`, `diag`는 실행 당시 사용한 이름이다. 아래 표에서 각 실행의 목적과 자료를 구분한다. Update 수는 optimizer가 가중치를 갱신한 횟수다.

| 폴더 | 실험 | 결과 | 저장한 파일 |
|---|---|---|---|
| `postfix/native-fp32` | **보완 후** 실제 135M, 2-GPU, FP32, 8 updates | 비교 통과 | rank0.json, rank1.json; 로그는 postfix/native-fp32.log |
| `postfix/hf-bf16` | **보완 후** 같은 HF backbone, 2-GPU, BF16, 8 updates | 비교 통과; 이전 실패 해소 | rank0.json, rank1.json; 로그는 postfix/hf-bf16.log |
| `postfix/native-bf16` | **보완 후** 서로 다른 Native/HF backbone, 2-GPU, BF16 | 비교 불일치; 해당 backbone 간 동등성은 보장 범위에서 제외 | rank0.json, rank1.json; 로그는 postfix/native-bf16.log |
| `postfix/native-moe-bf16-matrix` | **보완 후** Native MoE token GRPO, full·DP=2·TP=2 및 중단/재개 | 수치 비교·세 exact resumes 통과 | report.json, rank 결과; 로그는 postfix/native-moe-bf16-matrix.log |
| `cpu-final2` | 작은 76K 모델, CPU, FP32, 24 updates | 비교 통과 | rank0.json, run.log |
| `dp2-v1` | 같은 작은 모델, 2-GPU, FP32, 24 updates | 비교 통과 | rank0.json, rank1.json, run.log |
| `135m-dp2-fp32-v1` | 실제 135M 모델, 2-GPU, FP32, 8 updates | 비교 통과 | rank0.json, rank1.json, run.log |
| `135m-dp2-bf16-diag` | 실제 135M 모델, 2-GPU, BF16, 8 updates; 모든 update의 차이를 기록 | gradient/loss 비교 실패 | rank0.json, rank1.json, run.log |
| `135m-hf-dp2-bf16-v1` | 두 trainer에 같은 HF 모델 구현을 넣어 비교, 2-GPU, BF16, 8 updates | 여러 update 뒤 gradient/loss 비교 실패 | rank0.json, rank1.json, run.log |
| `hf-cpu-v3` | 위 HF 모델 연결 방식의 보조 검사, 작은 모델, CPU, 8 updates | 비교 통과 | rank0.json, run.log |
| `135m-dp2-bf16-v1` | BF16 비교의 첫 실행; 상세 기록 기능을 추가하기 전 | 학습은 완료했지만 첫 불일치에서 결과 비교 중단 | **run.log만 있음** |

`135m-dp2-bf16-v1/run.log`도 저장소에 포함한다. 처음 BF16 비교가 어디서 중단됐고 왜 상세 기록을 남기는 후속 실행을 했는지 추적하기 위한 기록이다. 보완 전 차이의 상세 값은 `135m-dp2-bf16-diag/rank0.json`, 최종 수정 효과는 **`postfix/hf-bf16/rank0.json`**에서 확인한다. 초기 실행을 다시 처리할 필요는 없다.

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

- `summary.json`: `final_decision`에 최종 판정·사용 조건·필수 수정 완료 여부, 나머지 항목에 각 실행의 판정과 최대 오차를 기록한 요약.
- `fp32_loss_trajectories.png` / `.svg`: FP32에서 두 trainer의 loss가 어떻게 변했는지 보여주는 그림.
- `precision_discrepancy.png` / `.svg`: FP32/BF16 비교의 update별 gradient 차이.
- `runtime.json`: 사용한 라이브러리 버전, GPU 범위, 실행 환경 설정.
- `source/initial`, `source/final`, `source/postfix`: 실험 도구와 trainer 코드의 세 버전. 각 폴더의 `source.tar.gz`는 코드 묶음, `source_hashes.json`은 코드 파일별 SHA256이다. `initial`은 최초 비교 코드, `final`은 상세 진단 코드, `postfix`는 gradient 반올림을 보완하고 회귀를 통과한 코드다. 각 결과의 script hash와 대조해 읽는다.
- `gradient_probe_before.json` / `gradient_probe_after.json`: 단일 BF16 logits에서 gradient를 독립 FP32 기준과 비교한 수정 전후 결과. 74개 불일치가 0개로 줄었다. `gradient_probe.py`에 같은 검사를 재현하는 코드가 있다.
- `manifest.json`: 이 폴더에 보관한 파일의 SHA256과 크기. 파일이 나중에 바뀌었는지 확인하는 용도다.

작은 모델은 스크립트의 seed=83과 Llama 설정으로 재구성한다. 135M 모델은 [모델 revision·파일 hash 기록](../2026-10-07-remediation/huggingface_manifest.json)에 지정한 SmolLM2 checkpoint를 사용한다. 큰 모델 weights와 update별 전체 optimizer 상태는 이 폴더에 보관하지 않는다. 다시 실행하는 명령은 [16번 노트의 재현 방법](../../16-grpo-trl-reference.md#재현과-블로그에서의-결론)에 있다.
