# CS336를 기준으로 검증하는 IronCore의 2-GPU 학습

이 노트는 `LanguageModelTrainer`(pretraining/SFT), `DPOTrainer`, `GRPOTrainer`를 실제 학습 도구로 검증하는 실험 기록이다. 기준은 [Stanford CS336 Spring 2026](https://cs336.stanford.edu/)의 구현·실험 주제이며, 과제 해답을 복제하거나 강의 전체를 이식했다고 주장하는 문서가 아니다. 강의의 요구 사항을 이 저장소의 코드 경로와 측정 가능한 검증 조건에 연결한다.

00–09는 최초 실험의 immutable 기록이다. **현재 구현의 보완 상태와 근거는 [10. Trainer 보완](10-trainer-remediation.md) 및 11–15 노트**를 기준으로 읽는다. 최초 matrix의 분산 동등성은 표준 AdamW oracle까지 입증하지 않았으며 후속 단계에서 독립 검증했다. EP의 최초 실패도 보존하고, 지원하는 2-GPU topology의 수정 후 결과를 별도로 기록한다.

## 읽는 순서

| 단계 | 노트 | 질문 | 근거 종류 |
|---|---|---|---|
| 00 | [환경·실험 계약](00-experiment-contract.md) | 무엇을 통과해야 학습 도구라고 판단하는가? | 버전, GPU, 데이터 hash, 허용 오차 |
| 01 | [학습 수치](01-training-numerics.md) | loss·gradient·optimizer update가 정의와 일치하는가? | 독립 기준과 회귀 테스트 |
| 02 | [데이터·SFT packing](02-data-and-packing.md) | target과 mask가 올바르며 문서 간 누출이 없는가? | label 정렬, causal mask, 분리 학습 비교 |
| 03 | [DP·TP·재개](03-distributed-and-resume.md) | GPU 수·microbatch 수를 바꾸어도 같은 학습인가? | NCCL, gradient/weight 비교, 중단·재개 |
| 04 | [실제 텍스트 학습](04-real-text-learning.md) | 실제 텍스트의 held-out loss가 줄어드는가? | 약 50M/130M, TinyStories |
| 05 | [3090 성능과 최적화](05-3090-systems.md) | 처리량과 메모리를 무엇이 제한하는가? | microbatch/context sweep, trace, telemetry |
| 06 | [SFT·DPO·GRPO](06-alignment.md) | 각 목적함수와 온라인 rollout이 올바른가? | reference, preference, reward, advantage |
| 07 | [강의 범위와 후속 실험](07-curriculum-coverage.md) | 어디까지 검증했고 무엇이 아직 필요한가? | scaling/data/kernel 평가의 미검증 범위 |
| 08 | [MoE 검증](08-moe.md) | routing·expert gradient와 분산 학습이 같은 목적함수를 따르는가? | 독립 수식, DP/TP, idle experts, EP 실패 oracle |
| 09 | [Profiler·HTML 보고서](09-profiler-html.md) | compute와 collective가 언제 겹치며 rank별로 어떻게 다른가? | 두 rank의 CUDA trace, interval union, interactive timeline |
| 10 | [보완 결과의 전체 상태](10-trainer-remediation.md) | 앞선 검증에서 남긴 취약점을 어떻게 보완했는가? | 단계별 gate와 결과 링크 |
| 11 | [수치·precision](11-numerics-and-precision.md) | optimizer와 global objective가 독립 기준과 맞는가? | 작은 gradient·불균일 batch·FP32 weights |
| 12 | [Stateful data·checkpoint](12-stateful-data-and-checkpoints.md) | prefetch·cursor·RNG·실패 처리를 복원하는가? | worker 0/2·atomic commit·checksum |
| 13 | [Alignment 보완](13-alignment-contracts.md) | GRPO/GSPO·behavior logp·실제 데이터가 맞는가? | math oracle·EOS·135M SFT/DPO/GRPO |
| 14 | [성능 보완](14-performance-remediation.md) | CE 재계산과 batched expert가 유용한가? | 3회 timing·VRAM·새 HTML profiler |
| 15 | [분산 학습 보완](15-distributed-training.md) | EP/FSDP/분산 optimizer를 사용할 수 있는가? | 네 trainer·native restart·rank fault injection |
| 16 | [GRPO 외부 trainer 대조](16-grpo-trl-reference.md) | TRL의 실제 학습 루프와 같은 update를 만드는가? | FP32 통과·BF16 strict gate 실패·통제된 135M 학습 |

[브라우저에서 여는 compute·collective 성능 보고서](profiling/profile_report.html)는 Dense DP=2, MoE DP=2, MoE TP=2의 두 GPU timeline을 담는다. 외부 서버 없이 run/rank 선택, 확대, category/phase 필터와 kernel 검색을 사용할 수 있다.

2026-10-07에 RTX 3090 두 장에서 실행했다. 네 trainer의 FP32/BF16 accumulation·DP=2·TP=2 비교가 통과했고, 총 56개 same-topology 재개 비교에서 최종 weights/loss 궤적이 정확히 일치했다. 실제 TinyStories 100-step pilot에서 52.8M/130.4M 모델의 held-out NLL은 각각 10.884→4.327, 10.940→3.654였다. 측정한 compile 대조에서는 130M 처리량이 13.7% 높아졌다.

각 노트의 `실측 결과`는 실제 실행 artifact만 근거로 한다. 계획·정적 분석·CPU 검증·CUDA 검증을 서로 대체하지 않는다. 전체 강의의 scaling law fitting, Common Crawl filtering/dedup, 자체 Triton attention, 장기 reasoning alignment의 품질 개선은 [07의 미검증 범위](07-curriculum-coverage.md)에 남겼다. [보존한 실험 근거](artifacts/2026-10-07/README.md)에서 raw JSON·CSV·실패 로그와 그래프를 확인할 수 있다.

## 실행 도구

- `scripts/validate_trainers.py`: 로컬 tokenizer와 통제된 데이터로 실제 trainer/model/optimizer/checkpoint를 실행한다. FP32 loss, clipping 이전 gradient, 최종 weights, rank 간 일치, reference 고정, scheduler/scaler와 재개 궤적을 비교한다.
- `scripts/benchmark_training.py`: 실제 텍스트의 train/valid token 파일을 준비하고 학습·메모리·동기화된 step 시간을 측정한다.
- `scripts/run_training_study.py`: 동일 GPU에서 한 작업씩 실행하여 model/microbatch/context/compile/recompute를 비교한다. OOM을 용량 한계로 기록하고 다른 오류는 중단한다.
- `scripts/summarize_training_study.py`: 결과·telemetry·수치 검증을 보존하고 PNG/SVG 및 측정 CSV를 만든다.

예시:

```bash
python scripts/validate_trainers.py --device cuda --architecture cs336 \
  --steps 8 --output /tmp/ironcore-correctness
python scripts/benchmark_training.py --prepare --data-dir /tmp/ironcore-corpus
CUDA_VISIBLE_DEVICES=0,1 python scripts/run_training_study.py \
  --data-dir /tmp/ironcore-corpus --output /tmp/ironcore-study \
  --wait-for-validation /tmp/ironcore-correctness/report.json
```

출력 디렉터리는 매 실행마다 새로 지정한다. artifact의 대형 weights/token 파일은 저장소에 넣지 않고, 결과 JSON·데이터 manifest·로그·trace의 요약과 hash를 `artifacts/`에 보존한다. `/tmp` 경로는 이번 세션의 작업 위치이며 장기 보관 위치는 아니다.

기본 CLI용 [50m systems YAML](../../configs/experiments/cs336_50m_systems_dp2.yaml)과 [130m systems YAML](../../configs/experiments/cs336_130m_systems_dp2.yaml)은 BF16/DP=2/microbatch 16/global batch 64 설정이다. 두 YAML의 모델·batch 구성을 유지한 4-step `ironcore train` smoke 실행이 통과했다. 이 설정은 random-token 시스템 workload이며 실제 텍스트 학습에는 위 benchmark 명령을 사용한다. CLI가 context manager를 사용하도록 수정하여 정상 종료 시 NCCL과 reward worker 자원도 해제한다.

## 강의 원문

확인일: 2026-10-07. 강의 사이트가 연결한 공개 과제 PDF의 본문 버전을 확인했다. GitHub README의 과거 연도 표기보다 PDF 버전을 우선한다.

| 과제 | 확인한 PDF 버전 | 이 노트의 연결 |
|---|---|---|
| [A1 Basics](https://github.com/stanford-cs336/assignment1-basics/blob/main/cs336_assignment1_basics.pdf) | 26.0.3 | tokenizer, Transformer 구성, optimizer, 작은 LM 학습 |
| [A2 Systems](https://github.com/stanford-cs336/assignment2-systems/blob/main/cs336_assignment2_systems.pdf) | 26.1.3 | profiling, mixed precision, checkpointing, kernels, 병렬 학습 |
| [A3 Scaling](https://github.com/stanford-cs336/assignment3-scaling/blob/main/cs336_assignment3_scaling.pdf) | 26.0.5 | isoFLOPs와 모델/데이터 예산 구분 |
| [A4 Data](https://github.com/stanford-cs336/assignment4-data/blob/main/cs336_assignment4_data.pdf) | 26.0.1 | 데이터 변환·필터·중복 제거 후 학습 품질 비교 |
| [A5 Alignment](https://github.com/stanford-cs336/assignment5-alignment/blob/main/cs336_spring2026_assignment5_alignment.pdf) | 26.0.0 | on-policy GRPO, sequence normalization, off-policy GRPO/GSPO |
| [A5 Safety supplement](https://github.com/stanford-cs336/assignment5-alignment/blob/main/cs336_spring2026_assignment5_supplement_safety_rlhf.pdf) | 26.0.0 | SFT, DPO |

## 보완 후 실행과 보고서

[최종 compute·collective HTML](profiling-remediation/profile_report.html)은 Dense CE, loop/batched MoE, EP=2의 4 workload와 두 rank를 담는다. [10–15 전체 결과](10-trainer-remediation.md)에 표준 optimizer·stateful data·실제 alignment·분산 도구의 지원 계약을 기록했다. 실제 GRPO reasoning 품질 개선은 입증하지 못했다.

검증한 systems preset은 [55M batched MoE DP=2](../../configs/experiments/cs336_55m_moe_batched_dp2.yaml) 및 [130M context8192 DP=2](../../configs/experiments/cs336_130m_long_context_dp2.yaml)이다. `CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 -m ironcore train --config <yaml>`로 실행한다. Preset은 mock data이며 실제-text 학습/평가는 `benchmark_training.py`, 135M alignment는 `benchmark_alignment.py`를 사용한다.
