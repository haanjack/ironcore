# 09. 두 GPU의 compute·collective profiler와 HTML 보고서

[인터랙티브 HTML 보고서](profiling/profile_report.html)는 외부 라이브러리나 서버 없이 브라우저에서 연다. Dense 52.8M DP=2, MoE 55.2M DP=2, MoE 55.2M TP=2의 두 rank를 실제로 수집했다. rank 선택, category/phase 필터, 1/5/20/100 ms 확대, 구간 이동, kernel 검색과 tooltip을 제공한다. 비교 표·수치·compact GPU events가 HTML 안에 있으므로 HTML 파일만 복사해도 실행된다. 블로그에 인용할 구조화된 값은 [profile_summary.json](profiling/profile_summary.json)에 보존한다.

## 실행 조건과 capture 검증

RTX 3090 × 2, NVLink NV4, GPU별 240 W cap, PyTorch 2.12.0+cu130, BF16, context 1024, microbatch 4, global batch 32다. 각 구성은 다른 GPU 작업 없이 순차 실행한 20-step pilot이다. DP=2는 accumulation 4, TP=2/DP=1은 accumulation 8이다. MoE는 routed 4/top-k 2/shared 1, EP=1, auxiliary alpha 0.01이다. 데이터와 optimizer는 [08](08-moe.md)의 TinyStories 실험을 따른다.

`--profile-ranks 0,1`로 CPU/CUDA activity, shapes, memory, Python stack을 수집한다. trainer warmup 10 updates를 지나 manager start=11/end=15, torch schedule wait=1/warmup=1/active=2/repeat=1을 사용한다. trainer는 update 종료 후 `step()`을 호출한다. 실제 trace의 CPU `training_update/N` 표식으로 확인한 capture는 **모든 구성·rank에서 update 13과 14**였다. [PyTorch profiler 문서](https://docs.pytorch.org/docs/2.12/profiler.html)는 schedule과 반복마다 호출하는 `step()`의 관계를 명시한다.

HTML 제작 중 기존 `ProfileManager.step()`이 manager 시작 전에도 torch profiler schedule을 진행시키는 오류를 발견했다. 이전 설정에서는 의도한 start=11 대신 update 3·4를 기록했다. `is_active`인 동안에만 schedule을 진행하도록 수정하고 시작 전/종료 후에 진행하지 않는 회귀 테스트를 추가했다. 수정 후 세 구성을 두 rank에서 다시 실행했다. [05](05-3090-systems.md), [08](08-moe.md)의 이전 단일-rank profile은 초반 capture의 kernel 증거로 유지하며, post-warmup timeline 근거는 이 문서를 사용한다. 이전 non-profiled 처리량은 이 schedule 오류의 영향을 받지 않는다.

## 시간의 정의

CUDA activity event의 `[start, end)`를 device별로 모으고 stream 간 겹침을 제거한다. `C`는 compute kernel 구간들의 합집합, `N`은 NCCL kernel 구간들의 합집합이다.

- Window: 첫 CUDA activity 시작부터 마지막 종료까지의 관측 구간.
- Compute / NCCL: 각각 `|C|`, `|N|`. 두 stream이 동시에 실행돼도 한 번만 센다.
- Overlap: `|C ∩ N|`. NCCL without compute: `|N \ C|`.
- Activity union: compute, NCCL, copy/memset을 모두 포함한 합집합.
- No recorded activity: window에서 activity union을 뺀 시간. GPU utilization이나 CPU 병목을 직접 측정한 값은 아니다.

Category/phase/kernel 표의 `Σ duration`은 event별 누적 시간이다. union과 의미가 다르며 다른 stream의 동시 실행을 중복 포함할 수 있다. CPU operator 시간은 inclusive이므로 부모/자식 operator의 시간도 더하지 않는다. 두 rank timestamp는 `baseTimeNanoseconds + ts × 1000`으로 같은 host 시간축에 정렬하고 nanosecond 정수로 계산한다. HTML에서는 큰 epoch 정수를 문자열로 저장하고 상대 ms로 그려 JavaScript 정밀도 손실을 피한다.

## 두 update에서 관측한 결과

아래는 device별 interval union이며 단위는 ms다. 서로 다른 run을 같은 wall-clock 효율로 해석하지 않는다.

| 구성 | rank | window | compute | NCCL | overlap | NCCL without compute | NCCL kernels |
|---|---:|---:|---:|---:|---:|---:|---:|
| Dense DP=2 | 0 | 476.92 | 293.39 | 6.91 | 3.48 | 3.43 | 14 |
| Dense DP=2 | 1 | 476.88 | 289.04 | 11.49 | 5.43 | 6.07 | 14 |
| MoE DP=2 | 0 | 1058.35 | 370.91 | 23.19 | 5.28 | 17.91 | 16 |
| MoE DP=2 | 1 | 1058.33 | 368.39 | 6.97 | 2.41 | 4.56 | 16 |
| MoE TP=2 | 0 | 2196.22 | 491.13 | 464.55 | 0.00 | 464.55 | 1750 |
| MoE TP=2 | 1 | 2196.24 | 488.03 | 328.23 | 0.00 | 328.23 | 1750 |

DP는 마지막 accumulated backward에서 gradient bucket을 줄이고, TP는 forward/backward의 projection·expert 경로에서 반복적으로 reduce한다. TP는 accumulation 횟수도 두 배여서 event 수의 차이를 단일 operation의 속도 차이로 보지 않는다. 모든 관측 NCCL kernels는 AllReduce로 분류됐다. TP의 CPU collective annotation은 rank당 1,756개로 kernel 수와 다르다. CPU annotation과 GPU annotation을 이중 집계하지 않는다.

이 capture에서 작은 MoE TP의 반복 collective와 compute overlap 부재는 실제 관측이다. 정확한 critical path, 통신을 없앴을 때의 speedup, SM occupancy, memory bandwidth는 이 trace만으로 정하지 않는다. NCCL duration은 peer 대기와 launch 순서, profiler의 rank별 영향까지 반영하므로 순수 전송 시간으로 읽지 않는다. collective에 모든 rank가 참여해야 하는 의미는 [NVIDIA NCCL 문서](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/collectives.html)를 따른다.

MoE DP의 kernel 수는 rank당 약 31,100개로 dense의 9,562개보다 많다. category 누적 routing/indexing 시간도 HTML에서 확인할 수 있다. 작은 expert GEMM, indexing, dispatch, CPU launch 비용이 MoE 성능을 제한한다는 해석은 후보 원인이다. grouped GEMM/fused dispatch의 isolated ablation으로 입증한 개선은 아니다. SDPA가 실제 FlashAttention kernel을 실행한 이름도 확인했다. Flash kernel template 안의 `cutlass::bfloat16_t`를 일반 GEMM으로 오분류하지 않도록 attention 규칙을 먼저 적용한다.

## phase와 communication payload

Forward, gradient norm/clip, optimizer에 CPU 표식을 넣었다. Backward는 Python autograd backward 범위로 잡는다. CUDA correlation ID로 kernel을 launch한 CPU runtime을 찾고, 같은 thread의 표식에 대응한다. autograd worker thread의 launch는 같은 process의 해당 시간 범위로 fallback한다. 이에 대응하지 않는 event는 Other/unknown으로 남긴다. GPU 실행 종료 시각을 CPU phase 끝에서 자르지 않으며 CPU collective annotation 자체를 compute kernel로 세지 않는다.

CPU NCCL annotation의 input dims/dtype으로 **logical input tensor bytes**를 추정한다. rank당 두 update의 합은 dense DP 약 201.40 MiB, MoE DP 약 210.45 MiB, MoE TP 약 5626.01 MiB다. 입력 tensor가 같아도 collective algorithm, rank 수, protocol에 따라 실제 이동 bytes는 달라진다. 이 값으로 NVLink bandwidth를 계산하지 않는다. input shape/type이 불완전하면 unknown으로 남긴다. [nccl-tests의 bandwidth 정의](https://github.com/NVIDIA/nccl-tests/blob/master/doc/PERFORMANCE.md)도 algorithm bandwidth와 bus bandwidth를 구분한다.

## 성능 baseline과 재현

HTML의 Steady tokens/s는 별도 non-profiled 실험의 초기 10 steps를 제외한 값이다. Dense DP=2 **172,872**, MoE DP=2 **127,318**, MoE TP=2 **69,530**이다. DP learning은 100 steps, TP pilot은 20 steps이므로 timing 길이도 다르다. Profile tokens/s는 instrumentation을 켠 run 전체 timing window의 값이며 active 두 update에만 해당하는 처리량이 아니다. profiler run의 VRAM peak도 instrumentation 영향을 포함한다. architecture와 microbatch auxiliary 목적함수의 차이가 있어 quality/FLOP 동등성은 주장하지 않는다.

```bash
CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 \
  scripts/benchmark_training.py --data-dir /tmp/ironcore-corpus \
  --output /tmp/profile-moe-dp2 --model-size 50m --moe \
  --context 1024 --micro-batch 4 --global-batch 32 --steps 20 \
  --profile --profile-ranks 0,1
```

Dense는 `--moe`를 생략하고 TP는 `--tp 2`를 추가한다. 별도 output directory를 사용한다. 보고서 생성:

```bash
python scripts/build_profile_report.py \
  --run-directory /tmp/profile-dense-dp2 \
  --run-directory /tmp/profile-moe-dp2 \
  --run-directory /tmp/profile-moe-tp2 \
  --baseline profile-dense-dp2=/tmp/dense-control \
  --baseline profile-moe-dp2=/tmp/moe-learning \
  --baseline profile-moe-tp2=/tmp/moe-tp2-study \
  --output docs/notes/profiling
```

[실제 실행 manifest](profiling/study.json), [실행 driver](profiling/commands.py.txt), [테스트 로그](profiling/regressions.log), [브라우저 검증](profiling/browser_validation.json), [최종 코드 hash](profiling/source_hashes.json)를 함께 보존한다. raw Chrome trace 약 1.8 GB는 임시 경로에 두고 각 trace의 SHA-256과 compact GPU timeline을 HTML/JSON에 남겼다. compact events에는 전체 CPU trace/stack/메모리 수명 정보가 없으므로 원본 trace를 대체하지 않는다. 전체 event를 장기 분석하려면 JSON의 raw path에 있는 원본도 별도 보관해야 한다.

CPU 회귀는 **80 passed**다. 검증은 interval union/overlap의 손계산 oracle, logical bytes, FlashAttention template 분류, launch-phase 대응, CPU/GPU annotation 중복 방지, profiler lifecycle을 포함한다. HTML은 desktop/mobile Chromium에서 모든 run/rank와 필터·검색·확대·이동을 확인한다. EP=2는 [08의 수치 실패](08-moe.md) 때문에 학습 경로에서 차단돼 있으며 이 보고서의 collectives는 EP=1 DP/TP 통신이다.
