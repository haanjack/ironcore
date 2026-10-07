# MoE 실험 근거

해석·명세·한계는 [08-moe.md](../../08-moe.md)를 따른다. 기존 dense archive를 덮어쓰지 않고 MoE 추가 실험을 보존한다.

- `ironcore-moe-fp32`, `ironcore-moe-bf16`: 네 trainer의 full/accum/DP/TP, reference, 재개, 실제 online GRPO reports와 rank별 summary.
- `ironcore-moe-idle`: bias margin으로 두 experts를 미선택 상태에 둔 pretraining 비교/재개.
- `ironcore-moe-aux-resume`: alpha=0.01, BF16 pretraining/SFT의 DP=2 exact resume.
- `ironcore-moe-baseline`, `tp_initial_failure.log`, `tp_initial_comparison.json`: SwiGLU TP 수정 전 실패. CPU layer tests가 통과해도 actual TP gradient가 틀린 경우다.
- `ironcore-moe-ep-oracle`: 정상 종료한 EP layer의 numerical failure 및 rank 1 DDP expert weight overwrite. `status: failed`를 그대로 보존한다. 별도 준비 실행의 인자 누락은 수치 증거로 사용하지 않는다.
- `study.json`, `measurements.csv`, `*_telemetry.csv`: 실제 learning/DP/TP/context run 설정과 rank별 loss/memory/timing/routing counts. 원본 study는 completed이며 profiler는 별도 파일이다.
- `profile_kernels.json`, `profile_*key_averages.csv`, `profile_results.json`: Chrome trace hash, kernel/CPU operator 요약, profiling run의 별도 결과. 이 run은 초반 update capture이며 성능 baseline에서 제외한다. schedule 수정 후 두 rank의 post-warmup timeline은 [09](../../09-profiler-html.md)와 [HTML](../../profiling/profile_report.html)을 따른다.
- `cli_smoke.json`, `cli_*_smoke.log`, `cli_ep_rejection.log`: 실제 2-GPU CLI 학습과 EP 사전 거부.
- `regressions.log`: 131 passed, 1 skipped. 기존 dense 회귀와 trainer/masking 테스트가 일부 중복되므로 단순 합산하지 않는다.
- `source_hashes.json`: MoE 실험 완료 후 working tree snapshot. run 시작별 immutable commit snapshot은 아니다. base commit과 dense stage 변경은 이전 archive도 함께 참조한다.
- `corpus_manifest.json`, `hardware.txt`, `environment_sitecustomize.txt`: 이전 단계와 동일한 corpus/hash, GPU/전력 조건, 이번 세션의 환경 workaround.
- `followup_commands.txt`, `final_gpu_commands.txt`: 실제 순차 실행 driver. `/tmp` 경로와 venv는 이번 세션의 위치다.
- `moe_learning_and_routing.png/.svg`: training objective에서 auxiliary loss를 뺀 NLL과 누적 expert 선택. smoothing 없음.
- `artifact_hashes.json`: README와 manifest 자체를 제외한 보존 파일의 SHA-256.

큰 model tensors, raw corpus/token files와 Chrome trace 원본은 repository에 넣지 않았다. `/tmp` 원본은 임시 저장 위치이며 장기 보관이 필요하면 별도 복사한다. trace hash/kernel 요약만으로 timeline 전체를 복원할 수는 없다. EP 실패를 MoE 전체의 실패나 EP=1 DP/TP의 통과로 대체하지 않는다.
