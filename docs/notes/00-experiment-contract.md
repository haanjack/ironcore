# 00. 환경과 검증 계약

## 실험의 주장

주장 A는 목적함수와 gradient가 올바르다는 것이다. 주장 B는 같은 초기 weights와 global batch를 사용했을 때 accumulation·DP·TP가 같은 update를 만든다는 것이다. 주장 C는 checkpoint 재개가 원래 학습 궤적을 이어 간다는 것이다. 주장 D는 실제 텍스트의 held-out loss가 개선된다는 것이다. 주장 E는 측정된 자원 조건에서 처리량을 높였다는 것이다. A–E에는 각각 별도 증거가 필요하다.

CS336의 기초 구현과 시스템 실험을 연결하는 실험 계약이다. [A1](https://github.com/stanford-cs336/assignment1-basics/blob/main/cs336_assignment1_basics.pdf), [A2](https://github.com/stanford-cs336/assignment2-systems/blob/main/cs336_assignment2_systems.pdf).

## 환경 확인

- 기반 commit: `6ca534b3a25280e530a6224ab990468d87ec3270`; 실험은 이 commit 위 작업 중인 수정본을 사용한다. 결과 보관 시 관련 파일의 SHA-256도 기록한다.
- GPU 0/1: NVIDIA GeForce RTX 3090, 각각 24,576 MiB. `nvidia-smi topo -m`은 두 카드 사이 `NV4`를 보고했다.
- 별도 GPU 2는 TITAN RTX다. 이번 GPU 실험은 `CUDA_VISIBLE_DEVICES=0,1`로 제한한다.
- 최초 확인 시 드라이버 590.48.01, 표시 CUDA 13.1; PyTorch `2.12.0+cu130`.
- 최초 측정한 power cap은 카드별 240 W다. 전력 제한·클럭 설정은 수정하지 않았다. 이 조건을 제외하고 다른 3090 결과와 속도를 직접 비교하지 않는다.
- 샌드박스에서는 GPU device가 가려져 `nvidia-smi`와 CUDA 확인이 실패했다. 샌드박스 밖 읽기 전용 확인에서 실제 접근이 정상임을 확인한 뒤 CUDA 실험을 실행했다.

## Python 환경의 재현성 문제

기존 환경에서 `torch._dynamo`가 Triton native library를 import할 때 segmentation fault가 발생했다. trainer 논리와 독립적인 환경 오류다. `/tmp/ironcore-validation-env`에 system-site-packages를 참조하는 별도 venv를 만들고 Triton 3.7.0 wheel을 재설치했다. 임시 `sitecustomize.py`에서 Triton을 먼저 import하고 TensorBoard의 TensorFlow stub을 사용하여 이 환경의 native import 순서 충돌을 우회했다. 저장소의 trainer에서 library를 mock한 것은 아니다.

일반 사용 환경에서는 정상 동작하는 CUDA PyTorch/Triton 조합이나 저장소의 컨테이너를 사용한다. 이번 workaround는 별도 artifact에 기록하며, 이를 일반적인 필수 설치법으로 취급하지 않는다. HF/Tiktoken/Matplotlib/compiler cache는 쓰기 가능한 `/tmp/ironcore-*`에 둔다.

## 통제된 수치 비교

| 항목 | 설정 |
|---|---|
| 기준 | 단일 GPU, global batch 8, microbatch 8, accumulation 1 |
| accumulation | 단일 GPU, microbatch 2, accumulation 4 |
| DP=2 | GPU별 microbatch 2, accumulation 2 |
| TP=2 | microbatch 2, accumulation 4, DP=1 |
| 초기화 | seed 42, dropout 0, FP32, TF32 비활성화, deterministic algorithms |
| 데이터 | 동일 global sample IDs와 labels; DP별 IDs는 서로 다른 stride |
| 수치 gate | FP32 tensor 비교 `atol=2e-5, rtol=2e-4`; loss도 같은 gate |
| 재개 gate | 같은 topology 안에서 weights·reference·후반 loss 궤적은 exact equality |
| gradient | 첫 update의 clipping 이전 gradient를 비교 |
| GRPO | global batch는 prompt 수; prompt당 4개 completion, 2 update epochs |

상대 오차만 사용하면 0에 가까운 원소에서 불안정하다. 절대·상대 tolerance를 함께 적용하고 최대 절대 오차도 보고한다. tolerance는 실측 뒤 임의로 늘리지 않는다. BF16 실험은 `atol=5e-3, rtol=5e-2`의 별도 수치 범위를 사용하며 FP32 검증의 대체 증거가 아니다. 실제 최종 matrix는 각 4 steps이며 장시간 안정성 실험과 구분한다.

이번 archive의 source hashes는 실행 완료 후 **최종 working tree**의 snapshot이다. 각 run 시작 시 모든 파일을 commit으로 고정한 snapshot은 아니며, 최초 실패→수정→재검증 과정에서 코드가 변했다. 기반 commit, 실패 로그, run별 설정과 마지막 source hash를 함께 보존하여 이 차이를 드러낸다. profiler/MFU/CLI 자원 해제 수정은 초기 learning pilot 이후 적용했다.

## 실제 학습과 성능의 계약

실제 텍스트 학습은 train/valid를 분리하고 tokenizer·token 파일 hash를 고정한다. 처리량 비교는 GPU 수, global batch, context, precision, attention backend, microbatch, warmup을 함께 기록한다. checkpoint/eval 시간과 정상 steady-state update 시간을 구분한다. wall-clock step time은 CUDA synchronize 후 측정하고 모든 rank 중 최댓값을 사용한다.

성능 실험은 한 번에 하나만 실행한다. OOM은 실패한 조합으로 기록하고 작은 조합에서 다시 시작한다. GPU utilization이 높다는 이유로 학습 품질이나 수치 정확성이 검증됐다고 판단하지 않는다.
