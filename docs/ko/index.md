# IronCore

**학습과 실험을 위한 LLM 훈련 프레임워크 (처음부터 구현)**

IronCore는 LLM 훈련 내부 구조를 직접 이해하기 위한 개인 프로젝트입니다. 분산 학습, 병렬화, 정렬(alignment), 최적화 등을 바닥부터 구현했습니다. NVIDIA Megatron-LM과 HuggingFace Transformers에서 영감을 받았습니다.

## 포함된 기능

- **훈련 모드:** 사전학습(pretrain), SFT, DPO, GRPO (Group Relative Policy Optimization)
- **병렬화:** 텐서 병렬(TP), 전문가 병렬(EP), 데이터 병렬(DP), 멀티노드, FSDP
- **모델 아키텍처:** 단일 `TransformerModel`로 GPT-2/3, LLaMA, Gemma, Qwen, Phi 지원
- **Mixture of Experts:** 부하 분산 + Z-loss, 전문가 병렬화
- **PEFT / LoRA:** TP-correct, 복제(replicated) 방식 어댑터 (샤딩 없음)
- **정렬 / RL:** 온라인 롤아웃, 그룹 상대적 어드밴티지, KL 패널티, 다중 백엔드 보상
- **옵티마이저:** Muon (Newton-Schulz) + AdamW 하이브리드; ZeRO-1 `DistributedOptimizer`
- **오프로드:** 옵티마이저 상태 오프로드, 가중치 스트리밍, 활성화 스필 (단일 GPU 데스크탑용)
- **체크포인팅:** 네이티브 (universal + 분산 TP) 및 HuggingFace 상호 운용

## 시작하는 방법

| 하고 싶은 것 | 문서 |
| --- | --- |
| 설치 및 첫 훈련 실행 | [시작하기](getting_started.md) |
| `ironcore` CLI 사용 | [CLI 가이드](cli_guide.md) · [CLI 레퍼런스](cli_reference.md) |
| TP/EP/DP/FSDP 이해 | [병렬화](parallelism.md) |
| DPO/GRPO 파인튜닝 | [정렬](alignment.md) · [보상 관리자](reward_manager.md) |
| VRAM 초과 모델 훈련 | [오프로드](offload.md) · [오프로드 설계](design/offload.md) |
| 설계 문서 읽기 | [설계 문서](design/index.md) |
| 기여 | [Contributing](https://github.com/haanjack/ironcore/blob/main/CONTRIBUTING.md) · [CI/CD 가이드](ci_cd_guide.md) |

> **하드웨어 참고.** IronCore는 듀얼 RTX 3090 (NVLink) 환경에서 개발·테스트되며,
> 완전한 기능을 위해 NGC PyTorch 컨테이너가 필요합니다 (FlashAttention이 베이스 이미지에 포함).
> 컨테이너 우선 설정은 [CONTRIBUTING.md](https://github.com/haanjack/ironcore/blob/main/CONTRIBUTING.md)를 참고하세요.
