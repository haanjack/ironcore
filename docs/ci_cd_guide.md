# CI/CD & Testing Guide

> Referenced by [CLAUDE.md](https://github.com/haanjack/ironcore/blob/main/CLAUDE.md) for detailed test setup and CI/CD workflow details.

## Overview

IronCore uses a resource-efficient CI/CD setup for a side project:

- **GitHub Actions (CPU)**: Automatic on every PR → Free validation
- **Self-hosted Runner (GPU)**: Optional; auto-runs on main if registered
- **Local Development**: Developer runs tests before PR

### Dual parallel execution

When you create a PR:
```
PR Created → [GitHub] CPU Tests ✅ + [Your Machine] GPU Tests ✅ (if runner online)
          → Both run in parallel, results shown on PR
```

---

## Quick Start for Developers

### Before Creating a PR

```bash
# CPU-only tests (always do this)
pytest tests/ -m "not cuda and not mp and not e2e and not hf_hub" -v

# If you have 2+ GPUs, also run single-GPU tests
pytest tests/ -m "cuda and not mp and not e2e" -v
```

### Create PR

```bash
git push origin feature/xyz

# GitHub Actions automatically:
# 1. Runs CPU tests on ubuntu-latest (GitHub-hosted)
# 2. Runs GPU tests on your machine (if runner registered & online)
```

---

## GitHub Actions Workflows

### Logic Tests (Every PR + push to main)

**Runner:** GitHub-hosted (ubuntu-latest)
**Command:** `pytest tests/ -m "not cuda and not mp and not e2e and not hf_hub"`
**Duration:** ~5-10 min | **Cost:** Free

### GPU Tests (Every PR + push to main)

**Runner:** `[self-hosted, gpu, mp]`
**Command:** `pytest tests/ -m "cuda and not mp and not e2e"`
**Duration:** ~10-15 min | **Cost:** Your hardware

### Distributed Tests (manual dispatch only — no runner currently serves it)

**Runner:** `[self-hosted, gpu, mp]`
**Command:** per-file `torchrun --nproc_per_node=2 -m pytest <file>` for all `mp`-marked files
**Duration:** ~10 min | **Note:** see "No multi-GPU runner" below

### E2E Tests (Manual dispatch only)

**Runner:** `[self-hosted, gpu, mp]`
**Command:** `pytest tests/ -m "e2e"`
**Duration:** ~10 min | **Note:** Tests self-spawn `torchrun` internally; triggered via `workflow_dispatch` with `test_mode=e2e`

---

## No multi-GPU runner

Nothing serves the `mp` pool right now, so `distributed-tests` and `e2e-tests`
only run on `workflow_dispatch` and will queue until an `mp` runner exists. The
2-GPU box was withdrawn because its memory is committed to other work; single-GPU
runners cannot stand in, because RCCL refuses two ranks on one device:

```
ncclInvalidUsage: Duplicate GPU detected : rank 0 and rank 1 both on CUDA device
```

They are left queueing rather than pointed at a single-GPU runner, where
`tests/conftest.py` would skip every `mp` test and the job would report green
having verified nothing. The multi-GPU suite is unverified, and the CI status
should say so.

### Running mp tests on one GPU, locally

gloo does allow two ranks to share a device, including allreduce on CUDA
tensors, so the suite can be partially exercised on a single-GPU host. This
found two real deadlocks (#104, #105) that only reproduced with two ranks.

Force gloo for both `init_process_group` and `new_group` — subgroups follow the
world group's backend as of #106, so `parallel.dist_backend: gloo` reaches them
— report two devices so the `device_count() < 2` guards pass, and pin both ranks
to `cuda:0`.

Coverage is partial. Of the 21 files in `DIST_TEST_FILES_NP2`:

| | files | what |
| --- | --- | --- |
| pass | 7 (30 tests) | TP attention, TP KV cache, LoRA TP, Muon TP/FSDP, DistributedOptimizer + checkpoint |
| fail | 8 | DDP/FSDP/offload paths — `ProcessGroupGloo` has no `perform_nocolor_split` |
| hang | 5 | EP, grad-norm, TP-equivalence |

This is a debugging technique, not a CI substitute: faking `device_count` can let
a test pass that would fail on two real devices, and gloo orders collectives
differently from NCCL.

## Setup Self-Hosted Runner (Optional)

Register your GPU machine so GitHub Actions automatically runs GPU tests.

### Runner labels and job routing

| Job | Runner labels | GPU requirement |
|---|---|---|
| `gpu-tests` (every PR) | `self-hosted, gpu` | 1+ GPU |
| `distributed-tests` (manual only) | `self-hosted, mp` | 2+ GPUs |
| `e2e-tests` (manual) | `self-hosted, mp` | 2+ GPUs |

The two labels name **job pools, not capabilities**, so a machine serves one or the
other and never both:

- `--labels gpu` — the single-GPU pool: PR `gpu-tests`.
- `--labels mp` — the multi-GPU pool: `distributed-tests` and `e2e-tests`.

Give a multi-GPU machine `mp` only. Labelling it `gpu,mp` also volunteers it for
every PR's single-GPU job, which is how `gpu-tests` ended up competing with other
work on the 2-GPU box and failing on OOM while a free single-GPU runner sat idle.

### Prerequisites

- Linux machine with at least 1 GPU
- GPU drivers and Docker with GPU passthrough installed

### Step 1: Generate Runner Token

1. Go to: `https://github.com/YOUR_USER/ironcore/settings/actions/runners`
2. Click **"New self-hosted runner"** → **Linux** → **x64**
3. Copy the `--token` value

### Step 2: Configure Runner

```bash
cd ~
mkdir -p github-runner && cd github-runner

# Download latest runner — check https://github.com/actions/runner/releases for latest version
RUNNER_VERSION="2.333.1"  # Update to latest if needed
curl -o actions-runner-linux-x64-${RUNNER_VERSION}.tar.gz \
  -L https://github.com/actions/runner/releases/download/v${RUNNER_VERSION}/actions-runner-linux-x64-${RUNNER_VERSION}.tar.gz
tar xzf ./actions-runner-linux-x64-${RUNNER_VERSION}.tar.gz

# Single-GPU machine (serves gpu-tests on PRs):
./config.sh \
  --url https://github.com/YOUR_USER/ironcore \
  --token NEW_TOKEN_HERE \
  --labels gpu \
  --unattended \
  --replace

# Multi-GPU machine (also serves distributed-tests and e2e-tests):
./config.sh \
  --url https://github.com/YOUR_USER/ironcore \
  --token NEW_TOKEN_HERE \
  --labels gpu,mp \
  --unattended \
  --replace
```

⚠️ **Important:**
1. Replace `YOUR_USER` with your GitHub username (e.g., `haanjack`)
2. Replace `NEW_TOKEN_HERE` with fresh token from Step 1 (tokens expire ~1 hour)
3. Check runner version: `https://github.com/actions/runner/releases`
4. Update `RUNNER_VERSION="X.X.X"` if newer available

### Step 3: Install as Service

```bash
sudo ./svc.sh install
sudo ./svc.sh start
sudo ./svc.sh status
```

### Step 4: Verify

Visit: `https://github.com/YOUR_USER/ironcore/settings/actions/runners`

Your machine should appear as **"Idle"** (green dot).

---

## Manual Trigger GPU Tests

If runner is registered but tests didn't auto-trigger:

**From GitHub UI:**
1. Go to **Actions** tab
2. Select **Tests** workflow → **Run workflow**
3. Choose `test_mode: gpu` → **Run**

**From CLI:**
```bash
gh workflow run test.yml -f test_mode=gpu
```

---

## Examples

### Feature A Development

```bash
git checkout -b feature/attention-opt
# ... code changes ...

pytest tests/ -m "not cuda and not mp and not e2e and not hf_hub"  # Test locally
git push origin feature/attention-opt    # Push → Auto PR tests (logic + GPU)
```

### Patch B (MP=2 Testing)

```bash
git checkout -b patch/tp-stability
# ... code changes + @pytest.mark.mp tests ...

pytest tests/ -m "cuda and not mp and not e2e" -v  # Single-GPU tests first
torchrun --nproc_per_node=2 -m pytest tests/multi_gpu/test_my_change.py -v  # Distributed
git push origin patch/tp-stability
```

---

## Troubleshooting

### GPU tests not running on PR

**Check:**
- Runner registered? → `https://github.com/YOUR_USER/ironcore/settings/actions/runners`
- Runner online? Check "Last seen" timestamp
- Restart runner: `sudo ~/github-runner/svc.sh restart`

### GPU tests skip with "GPU not available"

```bash
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, Devices: {torch.cuda.device_count()}')"
```

Fix: Install CUDA drivers (`nvidia-smi`) or update PyTorch: `pip install torch==2.x.x+cu118`

### Runner offline but need tests now

```bash
# Manual trigger
gh workflow run test.yml -f test_mode=gpu
# Will queue and wait for runner to come online
```

### Runner registration 404 error

Tokens expire about an hour after generation. If `config.sh` returns 404:

1. Generate a new token: go to your runner settings page, click "New self-hosted runner" → Linux → x64, and copy the fresh token.
2. Verify the URL is correct: `https://github.com/YOUR_USER/ironcore` (replace YOUR_USER).
3. Check network connectivity: `curl -I https://api.github.com`
4. Re-run `./config.sh` with the new token and correct URL.

---

## Test Configuration & Models

All test-specific configurations reside in `tests/fixtures/configs/` (separate from user-facing `configs/`).

### Test Models

**Qwen2.5-0.5B** (`tests/fixtures/configs/model/qwen2.5-0.5B.yaml`)
- Lightweight model for E2E smoke tests (GRPO, DPO)
- Requires: `Qwen/Qwen2.5-0.5B-Instruct` from HuggingFace (auto-downloaded on first use)
- Used in: `grpo_gsm8k_smoke_*.yaml` configs

### Test Dataset Configs

**GSM8K** (`tests/fixtures/configs/data/grpo_gsm8k.yaml`)
- Sample dataset for math reasoning GRPO training
- Used in smoke tests to validate reward manager and training loop
- Subset: ~100 examples (fast validation)

### Smoke Test Configs

Located in `tests/fixtures/configs/`:
- `grpo_gsm8k_smoke_fsdp.yaml` — FSDP distributed training (2-GPU)
- `grpo_gsm8k_smoke_rm_math.yaml` — Math reward function variant
- `grpo_gsm8k_smoke_rm_composite.yaml` — Composite reward (format + correctness)
- `grpo_gsm8k_smoke_rm.yaml` — Base reward config

**Usage in E2E Tests:**
```python
# tests/unit/reward/test_reward_manager.py
@pytest.mark.rlvr
@pytest.mark.e2e
@pytest.mark.mp
@pytest.mark.smoke
def test_reward_manager_config_trains():
    _run_training("tests/fixtures/configs/grpo_gsm8k_smoke_fsdp.yaml")
```

Run E2E smoke tests (opt-in):
```bash
pytest -m e2e tests/unit/reward/test_reward_manager.py -v
```

Note: `rlvr` alone also matches cheap GRPO math tests in `tests/unit/alignment/` which run in the default CPU tier. The `e2e` marker is the gate for expensive tests.

---

## Files & References

- [CLAUDE.md](https://github.com/haanjack/ironcore/blob/main/CLAUDE.md): Commands, architecture, test markers (for Claude)
- [tests/conftest.py](https://github.com/haanjack/ironcore/blob/main/tests/conftest.py): Auto-skip logic for cuda/mp markers (no marker registration — single source of truth is pyproject.toml)
- [.github/workflows/test.yml](https://github.com/haanjack/ironcore/blob/main/.github/workflows/test.yml): Workflow YAML definition
