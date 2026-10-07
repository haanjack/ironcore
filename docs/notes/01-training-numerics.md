# 01. loss·gradient·optimizer update

## 명세

[A1 Basics](https://github.com/stanford-cs336/assignment1-basics/blob/main/cs336_assignment1_basics.pdf)의 기초 구성과 학습 update를 연결한다. 기존 모델의 실행 가능성 외에, 정의한 목적함수에 대한 gradient와 optimizer update를 독립 기준으로 검증한다.

Pretraining의 token 평균은 `sum(mask * CE) / sum(mask)`다. 이 저장소의 SFT는 각 sample의 response token 평균을 구한 뒤 sample 간 평균을 구한다. 두 목적함수는 response 길이가 다르면 서로 다르다. 문서와 evaluation도 그 선택을 반영해야 한다.

동일 크기의 microbatch K개에 대한 sample 평균 loss는 각각 `loss/K`로 backward하면 전체 batch의 sample 평균과 같다. token 평균은 microbatch별 유효 token 수가 동일할 때 이 관계가 성립한다. masked token 수가 달라지는 pretraining batch나 크기가 다른 SFT packed row에서는 단순 microbatch 평균을 전체 token/sample 평균과 동일시할 수 없다.

## 발견과 수정

1. 모든 token이 masked인 pretraining loss의 분모가 0이었다. 분모를 최소 1로 제한하여 finite zero loss와 zero gradient를 만든다.
2. NaN/Inf loss 검사가 optimizer update 이후였다. 공통 accumulation 종료 시, update 전에 모든 rank가 nonfinite loss를 공유하고 중단하도록 변경했다.
3. GradScaler가 overflowing update를 건너뛰어도 LR scheduler가 진행했다. scale 감소로 skip을 확인하여 scheduler도 진행하지 않도록 변경했다. [PyTorch AMP 설명](https://docs.pytorch.org/docs/2.12/notes/amp_examples.html)의 step skip 동작을 기준으로 했다.
4. autocast에 `cuda:0` 같은 device string을 전달할 수 있었다. device type을 분리하고 FP32에서는 autocast를 사용하지 않는다.
5. trainer가 DDP wrapper의 `device` 속성에 의존하지 않도록 초기화 경로를 수정했다. CPU/Gloo DDP도 실제 공통 trainer 경로로 실행 가능하다.

## 실험과 결과

`tests/unit/trainers/test_training_correctness.py`는 실제 backward·AdamW·gradient clipping·CPU GradScaler를 사용한다. full-batch 기준 update를 별도로 계산하여 accumulation update와 비교한다. SFT response 길이를 다르게 만들어 sample 평균의 의미도 확인한다.

검증 항목: pretrain/SFT update 동등성, nonfinite loss에서 weights/optimizer/scheduler 불변, AMP overflow에서 update/scheduler skip 및 grad 제거, empty mask의 zero gradient, evaluation target 정렬과 objective 일치. chunked CE의 loss/gradient를 독립 `torch.nn.functional.cross_entropy`와 비교하는 6개 경우를 포함해 이 파일의 14개 검증이 통과했다.

확장 회귀 명령:

```bash
HF_HOME=/tmp/ironcore-hf python -m pytest \
  tests/unit/trainers tests/unit/alignment tests/unit/test_sft_masking.py \
  tests/unit/dataloader tests/unit/attention \
  tests/unit/optimizer/test_lr_scheduler.py tests/unit/parallel/test_grad_norm.py \
  tests/unit/checkpointing/test_universal_checkpoint_lora.py \
  tests/unit/parallel/test_tp_init_seed.py \
  tests/unit/parallel/test_comm_grad_correctness.py -q

HF_HOME=/tmp/ironcore-hf python -m pytest \
  tests/unit/profiler/test_profiler.py tests/unit/test_mfu.py -q
```

최종 core 회귀: **233 passed, 11 skipped**. 별도 profiler/MFU 회귀: **69 passed**. 합계 302 passed, 11 skipped이며 skip은 통과로 집계하지 않는다. [core 로그](artifacts/2026-10-07/regressions.log)와 [profiler/MFU 로그](artifacts/2026-10-07/profiler_mfu.log)를 보존했다. GPU/NCCL 증거는 pytest 수가 아니라 [03의 실제 trainer matrix](03-distributed-and-resume.md)로 판단한다.

## 해석의 한계

finite loss 자체가 모든 gradient의 유한성을 보장하지는 않는다. FP16 model parameter와 AMP의 조합, nonfinite gradient만 발생하는 경로, offload/FSDP/분산 optimizer 조합은 별도 gate가 필요하다. 이번 GPU 기준은 FP32/BF16과 standard optimizer, DP/TP다. 현재 pretraining의 global token 가중 평균은 동일 길이·동일 유효 token 수의 batch를 기준으로 검증한다.
