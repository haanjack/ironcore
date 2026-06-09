# 설계 문서

설계 문서는 서브시스템의 **아키텍처와 근거**를 담습니다. 왜(why)와 어떻게 맞는지(how it fits together)를 설명하며, 줄 단위의 구현 방법(how — 그건 코드)이나 사용 방법(how to use — 그건 [가이드](../index.md))은 다루지 않습니다. 서브시스템을 변경하기 전에 엔지니어가 읽는 지속적인 참고 문서입니다.

실례로서 [오프로드 시스템 설계](offload.md)를 참고하세요.

## 표준

각 설계 문서는 `docs/design/<subsystem>.md`에 위치하며 다음 골격을 따릅니다 (해당 없는 섹션은 생략):

| 섹션 | 목적 |
| --- | --- |
| **개요** | 한 문단: 서브시스템이 무엇을 하고 어떤 문제를 해결하는지. |
| **대상 / 제약 조건** | 설계가 최적화하는 하드웨어, 스케일, 가정. |
| **아키텍처** | 컴포넌트와 관계 — 다이어그램으로 시작. |
| **데이터 흐름** | 작업(순전파/역전파, 저장/로딩 등)별로 무엇이 어디로 이동하는지. |
| **컴포넌트 상호 작용** | 주요 시나리오에서 각 부분이 어떻게 결합되는지. |
| **설정** | 설정 인터페이스 (필드, 기본값, 검증 규칙). |
| **트레이드오프 / 알려진 병목** | 가능하면 측정된 정직한 한계. |
| **파일 인덱스** | 각 모듈/파일의 역할을 매핑하는 표. |

설계 문서는 검증 가능해야 합니다: 실제 클래스와 파일 경로를 명시하고, 성능 주장을 뒷받침하는 테스트나 벤치마크를 인용하세요.

## 다이어그램 규칙

구조가 비자명한 곳에는 다이어그램이 필요합니다. 두 가지 도구를 각각 가장 적합한 용도에 사용합니다:

| **Mermaid** (인라인)에 사용 | **Excalidraw** (에셋)에 사용 |
| --- | --- |
| 시퀀스, 흐름, 라이프사이클 | 메모리 / 공간적 레이아웃 |
| 의사결정 트리, 상태 머신 | "Hero" 아키텍처 다이어그램 |
| 의존성 / 관계 그래프 | 정밀한 위치 지정이나 주석이 필요한 경우 |

- **Mermaid**가 기본입니다. ` ```mermaid ` 코드 블록으로 인라인 작성 — MkDocs Material이 빌드 시 렌더링하며, git에서 diff가 깔끔합니다.
- **Excalidraw**는 Mermaid로 표현하기 어려운 다이어그램에 사용합니다. `.excalidraw` JSON 소스와 내보낸 `.png` 모두 `docs/design/assets/`에 커밋하고 PNG를 임베드합니다. JSON이 편집 가능한 소스, PNG는 사이트에 표시되는 것입니다.

```text
docs/design/
├── index.md            # 이 표준
├── <subsystem>.md      # 서브시스템당 설계 문서 하나
└── assets/
    ├── <name>.excalidraw   # 편집 가능한 소스
    └── <name>.png          # 렌더링된 것, .md에 임베드
```

### Excalidraw 렌더링

`excalidraw-diagram` 스킬이 JSON을 생성하고 PNG로 렌더링합니다 (Playwright + headless Chromium). 일회성 설정:

```bash
cd ~/.claude/skills/excalidraw-diagram/references
uv sync && uv run playwright install chromium
```

다이어그램 편집 후 렌더링:

```bash
uv run python ~/.claude/skills/excalidraw-diagram/references/render_excalidraw.py \
    docs/design/assets/<name>.excalidraw -o docs/design/assets/<name>.png
```
