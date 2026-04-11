# CI/CD & Testing Guide

> Referenced by [CLAUDE.md](../CLAUDE.md) for detailed test setup and CI/CD workflow details.

## Overview

IronCore uses a resource-efficient CI/CD setup for a side project:

- **GitHub Actions (CPU)**: Automatic on every PR → Free validation
- **Self-hosted Runner (GPU)**: Optional; auto-runs on main if registered
- **Local Development**: Developer runs tests before PR

### Dual Parallel Execution

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
pytest tests/ -m "not cuda and not mp" -v

# If you have 2+ GPUs, also run GPU tests
pytest tests/ -m "cuda or mp" -v
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

### CPU Tests (Automatic)

**Triggers:** Every PR, push to main
**Runner:** GitHub-hosted (ubuntu-latest)
**Command:** `pytest tests/ -m "not cuda and not mp"`
**Duration:** ~5-10 min | **Cost:** Free

### GPU Tests (Optional with Self-hosted Runner)

**Triggers:** Push to main, manual workflow_dispatch
**Runner:** `[self-hosted, gpu, mp]`
**Command:** `pytest tests/ -m "cuda or mp"`
**Duration:** ~10-20 min | **Cost:** Your hardware

---

## Setup Self-Hosted Runner (Optional)

Register your GPU machine so GitHub Actions automatically runs GPU tests.

### Prerequisites

- Linux machine with 2+ NVIDIA GPUs
- CUDA drivers installed
- PyTorch with GPU support

### Step 1: Generate Runner Token

1. Go to: `https://github.com/YOUR_USER/ironcore/settings/actions/runners`
2. Click **"New self-hosted runner"** → **Linux** → **x64**
3. Copy the `--token` value

### Step 2: Configure Runner

```bash
cd ~
mkdir -p github-runner && cd github-runner

# Download runner
curl -o actions-runner-linux-x64-2.333.1.tar.gz \
  -L https://github.com/actions/runner/releases/download/v2.333.1/actions-runner-linux-x64-2.333.1.tar.gz
tar xzf ./actions-runner-linux-x64-2.333.1.tar.gz

# Configure
./config.sh \
  --url https://github.com/YOUR_USER/ironcore \
  --token YOUR_TOKEN_HERE \
  --labels gpu,mp \
  --unattended \
  --replace
```

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

pytest tests/ -m "not cuda and not mp"  # Test locally
git push origin feature/attention-opt    # Push → Auto PR tests
```

### Patch B (MP=2 Testing)

```bash
git checkout -b patch/tp-stability
# ... code changes + @pytest.mark.mp tests ...

pytest tests/ -m "mp" -v                # Test if 2+ GPUs available
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

---

## Files & References

- [CLAUDE.md](../CLAUDE.md): Commands, architecture, test markers (for Claude)
- [tests/conftest.py](../tests/conftest.py): Marker definitions and auto-skip logic
- [.github/workflows/test.yml](../.github/workflows/test.yml): Workflow YAML definition
