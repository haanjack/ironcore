# GRPO / TRL 기준 구현 대조

[16 노트](../../16-grpo-trl-reference.md)의 원본 결과다. `cpu-final2`, `dp2-v1`, `135m-dp2-fp32-v1`의 FP32 비교는 통과했다. `135m-dp2-bf16-diag`와 `135m-hf-dp2-bf16-v1`은 strict tensor/loss gate 실패를 담는다. Failure JSON과 nonzero-exit 로그를 성공으로 덮어쓰지 않는다. `135m-dp2-bf16-v1`은 처음 tensor에서 중단한 초기 실패 로그다.

`rank*.json`에는 optimizer step별 전체 trainable tensor의 최대 error, loss curve, initial/frozen-reference exact 검사, 후보 확률, 실행 명령·script SHA256·TRL source SHA256이 있다. `runtime.json`에 package versions와 비교의 계약을 기록한다. Large 135M weights와 optimizer snapshots는 저장소에 포함하지 않는다. `source_hashes.json`과 portable source archive는 결과를 만든 코드의 provenance이며 학습의 inputs/outputs를 대체하지 않는다.

`source/initial`은 첫 FP32·BF16 gate의 source, `source/final`은 failure를 끝까지 기록하고 같은 HF backbone 진단을 지원하는 source다. 실패 진단 중 source가 달랐던 report의 script hash도 그대로 보존한다. Synthetic 초기 모델은 script의 seed=83·HF Llama config로 재구성한다. 실제 135M은 이전 remediation의 `huggingface_manifest.json`에 고정된 SmolLM2 revision을 사용한다.

`manifest.json`은 이 디렉터리의 각 파일 SHA256과 크기를 기록한다. `summary.json`은 successful/failed gate index다. Scripted generation의 reward 평균은 고정되어 있으므로 후보 확률·update 방향을 생성 성공률이나 held-out reasoning 개선으로 해석하지 않는다.
