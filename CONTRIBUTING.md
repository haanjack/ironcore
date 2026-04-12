# Contributing to IronCore

Thank you for contributing. This document is the single source of truth for how to set up the development environment, write code, test changes, reason about performance, and submit a PR. Read it end-to-end before starting work.

---

## Table of Contents

1. [Development Setup](#1-development-setup)
2. [Coding Standards](#2-coding-standards)
3. [Testing Policy](#3-testing-policy)
4. [Performance Guide](#4-performance-guide)
5. [Development Workflow](#5-development-workflow)
6. [PR Description Template](#6-pr-description-template)

---

## 1. Development Setup

### Container first

IronCore requires a GPU container for all non-trivial development. **Do not attempt to install flash-attn via pip on the host.** Flash attention is provided by the NGC PyTorch base image (`nvcr.io/nvidia/pytorch:25.12-py3`) and requires a matching CUDA toolchain to build. The container ships a pre-built version; there is no host pip path that reliably works.

All integration and multi-GPU tests must be run inside the container.

### Prerequisites

- Docker with [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) (or ROCm equivalent for AMD GPUs)
- A `.env` file in the repo root with at minimum:

```bash
# Required: paths visible inside the container
DATASET_DIR=/path/to/datasets
MODEL_DIR=/path/to/models

# Optional: change the NGC version or target ROCm
# NGC_VERSION=25.12
# ARCH=rocm
```

### Build the container

```bash
# CUDA (default)
./scripts/docker/build.sh

# ROCm
ARCH=rocm ./scripts/docker/build.sh
```

This produces `ironcore:cuda` or `ironcore:rocm`.

### Run the container

```bash
# Interactive shell
./scripts/docker/launch.sh bash

# One-shot command (e.g., run tests)
./scripts/docker/launch.sh pytest tests/

# Full test suite
./scripts/docker/launch.sh ./scripts/run_tests.sh
```

The repo root is mounted at `/workspace` inside the container. The working directory is `/workspace` when the container starts.

### Install the package inside the container

```bash
# Inside the container, at /workspace
pip install -e .[dev]
```

This installs IronCore in editable mode plus `pytest` and `ruff` (the `dev` extras). The package is now importable as `ironcore`.

### Dependency management

| Where | What goes here |
|---|---|
| `pyproject.toml` `dependencies` | Runtime Python packages needed to run IronCore |
| `pyproject.toml` `[dev]` optional-dependencies | `pytest`, `ruff`, and other dev tools |
| `requirements.txt` | Mirror of `pyproject.toml` dependencies (kept in sync) |
| NGC base image | Flash-attn, cuDNN, NCCL, PyTorch — **do not add these to pyproject.toml** |

To upgrade PyTorch or flash-attn, bump `NGC_VERSION` in `.env` (or the `Dockerfile` `ARG BASE_IMAGE`) and rebuild the container. Do not pin these packages inside the repo.

---

## 2. Coding Standards

### Lint before every commit

CI will reject PRs that fail linting. Run this before committing:

```bash
ruff check ironcore/ tests/        # check for errors
ruff format ironcore/ tests/       # auto-format
ruff check --fix ironcore/ tests/  # auto-fix safe fixable errors
```

The GitHub Actions workflow (`.github/workflows/lint.yml`) runs `ruff check` and `ruff format --check` on every push to `main` and every PR.

### Type hints

Every new public function and method signature must include type annotations on all parameters and the return type. Use `from __future__ import annotations` at the top of new modules so PEP 604 union syntax (`X | Y`) works on Python 3.10+:

```python
from __future__ import annotations


def compute_loss(logits: torch.Tensor, labels: torch.Tensor, beta: float = 0.1) -> torch.Tensor:
    ...
```

Follow the style of `ironcore/utils/mfu.py` for dataclass-based APIs.

### Language

All code, comments, docstrings, commit messages, and PR descriptions must be written in **English**.

### Formatting

Formatting is defined in `pyproject.toml` under `[tool.ruff]` and `[tool.ruff.format]`. The canonical settings are:

| Setting | Value |
|---|---|
| Line length | 100 |
| Target Python | 3.12 |
| Quote style | Double |
| Indent style | Space |

Run `ruff format` — do not hand-format. Do not add per-file `# noqa` overrides or `.ruff.toml` overrides without discussion.

### Naming

- Functions and variables: `snake_case`
- Classes: `PascalCase`
- Constants: `UPPER_SNAKE_CASE`
- Module-level private helpers: `_single_leading_underscore`

Ruff's `N` ruleset enforces most of this. The existing `N806`, `N812`, and `N817` exemptions are intentional — do not remove them.

---

## 3. Testing Policy

### Directory structure

Tests mirror the source tree under the appropriate tier:

```
tests/
├── unit/           # CPU-only logic tests (no GPU required)
├── integration/    # Multi-component tests (requires GPU via torchrun)
├── multi_gpu/      # Distributed tests (2+ GPUs, torchrun)
├── regression/     # Bug-fix pinning tests
└── property/       # Invariant / property-based tests
```

Place new tests in the tier that matches the scope of the change. A change to `ironcore/optimizer/muon.py` gets unit tests in `tests/unit/optimizer/test_muon_*.py`. See `tests/test_suite.md` for the full directory structure and fixture inventory.

### Pytest markers

Use only **registered markers** (defined in `pyproject.toml`):

| Marker | When to use |
|---|---|
| `@pytest.mark.unit` | Fast, CPU-only logic tests |
| `@pytest.mark.integration` | Multi-component tests |
| `@pytest.mark.performance` | Benchmark / stress tests |
| `@pytest.mark.slow` | Long-running logic tests (>1 s, no GPU) |
| `@pytest.mark.tp1` | Tensor parallel tests with TP=1 |
| `@pytest.mark.tp2` | Tensor parallel tests with TP=2 |
| `@pytest.mark.cuda` | Tests requiring a single CUDA GPU |
| `@pytest.mark.kv_cache` | KV cache related tests |
| `@pytest.mark.paged_attention` | Paged attention tests |
| `@pytest.mark.prefix_cache` | Prefix cache tests |
| `@pytest.mark.rlvr` | E2E GRPO smoke tests (2 GPUs, ~10 min, opt-in) |

**Adding a new marker** requires registering it in `pyproject.toml` `[tool.pytest.ini_options] markers` in the same PR.

> **Known issue:** `tests/test_suite.md` references an `mp` (multi-process) marker that is not currently registered in `pyproject.toml`. Until this is resolved, use the `tests/multi_gpu/` directory convention rather than an `mp` marker.

### Regression tests for new layers

Every new `nn.Module` added to `ironcore/layers/` or `ironcore/models/` must include a regression test that pins **both forward output values and backward gradient values** against a fixed-seed reference. Place these in `tests/regression/test_<layer>_values.py`.

```python
# Example pattern
def test_my_layer_forward_regression():
    torch.manual_seed(42)
    layer = MyLayer(config)
    x = torch.randn(2, 16, 64)
    out = layer(x)
    # Pin with torch.testing.assert_close or hard-coded expected
    assert out.shape == (2, 16, 64)
    torch.testing.assert_close(out.mean(), torch.tensor(-0.0123), atol=1e-4, rtol=0)

def test_my_layer_backward_regression():
    torch.manual_seed(42)
    layer = MyLayer(config)
    x = torch.randn(2, 16, 64, requires_grad=True)
    out = layer(x)
    out.sum().backward()
    torch.testing.assert_close(x.grad.norm(), torch.tensor(3.456), atol=1e-3, rtol=0)
```

This catches silent numerical drift from refactors.

### Use IronCore trainers and inference — do not reimplement

Tests must drive training through `BaseTrainer` / `LanguageModelTrainer` / `DPOTrainer` / `GRPOTrainer`, and inference through `LanguageModel.generate()`. Do **not** reimplement a training loop, gradient accumulation loop, or generation loop inside a test file.

Use the shared config builders from `tests/fixtures/config_fixtures.py`:

```python
from tests.fixtures.config_fixtures import create_small_test_config

def test_training_step(create_small_test_config):
    config = create_small_test_config()
    # drive training through the trainer, not manually
```

### Unit tests for logic changes

Any change to pure-Python logic — config validation, loss math, collator behavior, optimizer update rules — must have a `tests/unit/` test that runs without GPU and is covered by the default `pytest tests/` run.

### Multi-GPU tests for parallelism changes

Any change touching `ironcore/parallel/`, TP/DP/EP communication, FSDP wrapping, or `DistributedOptimizer` must add or update a test in `tests/multi_gpu/` that exercises the distributed path:

```bash
torchrun --nproc_per_node=2 -m pytest tests/multi_gpu/test_my_change.py -v
```

### Running tests

```bash
# Default: unit + regression (CPU OK, ~5 min)
pytest tests/

# Single-GPU tests
pytest tests/ -m "cuda or tp1"

# Full suite including multi-GPU (run inside container)
./scripts/run_tests.sh

# Skip multi-GPU (faster local loop)
./scripts/run_tests.sh --quick

# E2E GRPO smoke tests (opt-in, 2 GPUs, ~10 min)
./scripts/run_tests.sh --rlvr
```

---

## 4. Performance Guide

### Memory efficiency

When adding or modifying tensor operations, audit for redundant copies:

- Avoid unnecessary `.clone()`, `.contiguous()`, or `.detach().clone()` calls in hot paths.
- Prefer in-place operations where the semantics are safe (e.g., `add_`, `mul_`, `copy_`).
- Use `.view()` or `.reshape()` over `.clone()` + reshape when no copy is needed.
- Free temporary tensors early with `del tensor; torch.cuda.empty_cache()` in memory-critical loops.

For changes that touch hot paths (forward/backward, KV cache update, MoE dispatch/gather), include a **before/after memory report** in the PR description using:

```python
from ironcore.utils.memory import get_detailed_memory_breakdown, format_memory_report

breakdown = get_detailed_memory_breakdown(model, optimizer)
print(format_memory_report(breakdown, title="After change"))
```

Run this on a reproducible config (e.g., `tests/fixtures/configs/model/qwen2.5-0.5B.yaml`, batch size 4, seq_len 1024).

### MFU alignment

Any change to layer compute — attention kernels, MLP, MoE routing/expert forward, embeddings — must include an **MFU estimate** in the PR description.

Use `ironcore.utils.mfu.MFUCalculator`:

```python
from ironcore.utils.mfu import MFUCalculator
from ironcore.config import ModelConfig

cfg = ModelConfig(...)  # or load from YAML
calc = MFUCalculator.from_config(cfg)
result = calc.compute_tflops(batch_size=4, seq_len=1024, step_time_seconds=0.123)
print(result)  # e.g. "12.34 TFLOPS/s/GPU | 4,096 tok/step"
```

Report measured TFLOPS/s/GPU before and after on a fixed config. A bare number is not enough — **explain why MFU changed**: which operations were added or removed, what memory traffic changed, and whether a kernel fusion was applied or broken.

---

## 5. Development Workflow

### Start from main

Always branch from an up-to-date `main`:

```bash
git fetch origin
git checkout -b <type>/<short-slug> origin/main
```

### Branch naming

| Prefix | Use for |
|---|---|
| `feature/<slug>` | New functionality |
| `fix/<slug>` | Bug fixes |
| `perf/<slug>` | Performance improvements (no behavior change) |
| `refactor/<slug>` | Internal restructuring (no behavior change) |
| `docs/<slug>` | Documentation only |
| `test/<slug>` | Test-only changes |
| `ci/<slug>` | CI, build scripts, tooling |

### Commit message convention

Follow [Conventional Commits](https://www.conventionalcommits.org/):

```
<type>(<scope>): <subject>
```

**Type** is one of: `feat`, `fix`, `perf`, `refactor`, `docs`, `test`, `ci`

**Scope** is one of the IronCore subpackages — use the actual directory name so scope tags are greppable:

`alignment` | `checkpointing` | `cli` | `config` | `dataloader` | `eval` | `layers` | `models` | `optimizer` | `parallel` | `peft` | `preprocessing` | `tokenizer` | `trainers` | `utils`

Use `*` for cross-cutting changes that touch multiple subsystems.

**Examples:**

```
feat(optimizer): add Lion optimizer with decoupled weight decay
fix(parallel): correct TP all-gather shape in GQA attention
perf(layers): fuse RMSNorm with linear projection
refactor(alignment): extract rollout buffer into separate module
docs(*): add CONTRIBUTING.md
test(dataloader): add regression test for bin-packing edge cases
ci(*): add GPU smoke test to weekly workflow
```

### PR title

Use the same `<type>(<scope>): <subject>` format as the commit subject.

### PR target

All PRs target `main`. Direct pushes to `main` and force-pushes to shared branches are not permitted.

### PR description

Use the template in [Section 6](#6-pr-description-template). The minimum required sections are **Summary** and **Test plan**. The MFU and Memory sections are required only when the change touches layer compute or hot-path tensor ops respectively.

**For feature PRs:** If you add new user-facing capabilities (e.g., new training mode, optimizer, loss function, logger, evaluator), update the Features list in `README.md` to document the new capability concisely.

### Before opening a PR

1. `ruff check ironcore/ tests/ && ruff format --check ironcore/ tests/` — must be clean.
2. `pytest tests/` — must pass with 0 failures.
3. If GPU changes: run the relevant integration tier inside the container.
4. Fill out the PR description template completely.

---

## 6. PR Description Template

Copy and fill in this template when opening a PR. Delete sections that do not apply (MFU/Memory are optional for non-compute changes).

```markdown
## Summary

- <!-- What changed and why — 1–3 bullets -->
- 
- 

## Test plan

Commands run and results:

```bash
# Example
pytest tests/unit/optimizer/ -v
# 23 passed, 0 failed
```

- [ ] Unit tests pass: `pytest tests/`
- [ ] Lint passes: `ruff check ironcore/ tests/ && ruff format --check ironcore/ tests/`
- [ ] Integration tests pass (if GPU change): `./scripts/run_tests.sh --quick`
- [ ] Multi-GPU tests pass (if parallelism change): `torchrun --nproc_per_node=2 -m pytest tests/multi_gpu/ -v`

## MFU impact

<!-- Required for changes to layer compute (attention, MLP, MoE, embeddings). -->
<!-- Use MFUCalculator on tests/fixtures/configs/model/qwen2.5-0.5B.yaml, batch 4, seq_len 1024 -->

| | Before | After |
|---|---|---|
| TFLOPS/s/GPU | | |
| Tokens/step | | |

Explanation: <!-- Why did MFU change? Extra FLOPs, memory traffic, kernel fusion? -->

## Memory impact

<!-- Required for hot-path changes (forward/backward, KV cache, MoE dispatch). -->
<!-- Use get_detailed_memory_breakdown(model, optimizer) on the same fixed config. -->

| Component | Before (MiB) | After (MiB) |
|---|---|---|
| Parameters | | |
| Gradients | | |
| Optimizer states | | |
| Activations | | |

## Breaking changes

<!-- List any config schema changes, public API changes, or checkpoint format changes. -->
<!-- Delete this section if none. -->
```
