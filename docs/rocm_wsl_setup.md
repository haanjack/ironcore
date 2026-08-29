# ROCm on WSL2 — AMD Strix Halo (gfx1151)

Verified working on 2026-07-31 with an AMD Ryzen AI MAX+ 395 (Radeon 8060S iGPU, gfx1151)
under Windows 11 + WSL2 (Ubuntu 24.04, kernel 6.18.33.2-microsoft-standard-WSL2).

## Why WSL needs an extra piece

Under WSL2 there is no `amdgpu` kernel driver and therefore **no `/dev/kfd`** — the device
node the ROCm runtime normally talks to. Instead the GPU is reached through Microsoft's
DXCore paravirtualization layer at `/dev/dxg`.

The bridge between the two is **ROCDXG** (`librocdxg`), a user-mode shim that the ROCr
runtime `dlopen`s. Without it, a stock `rocm/pytorch` image fails at HSA init:

```
WSL environment detected.
hsa api call failure ... HSA_STATUS_ERROR_OUT_OF_RESOURCES
```

`librocdxg` is self-contained (a ~185 KB `.deb`), so **nothing needs to be installed in the
WSL distro** — it is baked into the image. Only `libdxcore.so` comes from the host, because
the Windows driver supplies it and it cannot be shipped in a layer.

Strix Halo support: ROCDXG **1.2.0+** covers Ryzen AI Max+ 395; this repo defaults to 1.2.1.

## Requirements

- Windows 11 with a current AMD Adrenalin driver (≥ 26.2.2). Verified here with 32.0.23033.5002.
- WSL2 with `/dev/dxg` present and `/usr/lib/wsl/lib/libdxcore.so` available.
- Docker Engine inside WSL.

Quick check:

```bash
ls /dev/dxg /usr/lib/wsl/lib/libdxcore.so
```

## Build and run

```bash
ARCH=rocm-wsl ./scripts/docker/build.sh                       # bakes in ROCDXG
ARCH=rocm-wsl ./scripts/docker/launch.sh                      # interactive shell
ARCH=rocm-wsl ./scripts/docker/launch.sh ./ironcore-run train --config configs/example.yaml
```

`ARCH` is auto-detected from `/dev/dxg`, so it can usually be omitted.

Override the base image or the ROCDXG version if needed:

```bash
ROCM_IMAGE=rocm/pytorch:rocm7.14_ubuntu24.04_py3.12_pytorch_release_2.12.0 \
ROCDXG_VERSION=1.2.1 ARCH=rocm-wsl ./scripts/docker/build.sh
```

If you just added yourself to the `docker` group, your current shell has stale group
membership — either start a new login shell or prefix commands with `/usr/bin/sg docker -c "…"`.

## Verifying the GPU

```bash
ARCH=rocm-wsl ./scripts/docker/launch.sh python -c "
import torch
p = torch.cuda.get_device_properties(0)
print(torch.cuda.is_available(), p.name, p.gcnArchName, round(p.total_memory/1024**3,1), 'GiB')"
```

Observed on this machine:

```
True  AMD Radeon(TM) 8060S Graphics  gfx1151  102.1 GiB
```

Measured throughput (ROCm 7.2.4 / PyTorch 2.10, after warm-up):

| Workload | Result |
| --- | --- |
| bf16 4096³ matmul | 8.4 ms → 16.3 TFLOP/s |
| fp16 4096³ matmul | 8.7 ms → 15.8 TFLOP/s |
| fp32 4096³ matmul | 100 ms → 1.4 TFLOP/s |
| SDPA (2×8×1024×64, bf16) | 2.4 ms |
| `configs/example.yaml` (GPT-2 small, bf16) | ~6–9k tok/s, 5–7 TFLOPS/s/GPU |

Always discard the first iterations when benchmarking — kernel JIT makes an unwarmed
matmul look ~45× slower than it is.

## Things that differ from a CUDA box

**Attention.** `flash-attn` has no working gfx1151 build; the config flag `use_flash_attn`
degrades gracefully because `layers/attention.py` falls back to SDPA when the import fails.
SDPA itself is fast only when the AOTriton flash backend is enabled — PyTorch ≤ 2.12 ships an
AOTriton that still flags gfx1151 as experimental, so `launch.sh` sets
`TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1`. Without it SDPA silently uses the math backend
(~19× slower). Head dim must be ≤ 256.

**bf16 stability.** gfx1151 has documented bf16 NaN patterns upstream (ROCm/ROCm#6034):
very small total batch sizes, `head_dim=32`, and high learning rates have all reproduced NaNs.
Nothing appeared in the smoke run, but if a training diverges early, test fp32 or fp16 before
assuming the bug is in IronCore.

**Single GPU only.** AMD does not support multi-GPU on WSL, so TP/EP/DP stay at 1 and `gloo`
is the correct distributed backend. `initialize_parallelism()` correctly no-ops here
("Skipping parallelism wrapping (single GPU, no FSDP)").

**Tooling.** `rocm-smi` does not work under WSL; use `amd-smi` (limited) or `torch.cuda.*`.
ROCm profiler and debugger are unsupported.

**Memory.** The 102 GiB figure is unified memory, not dedicated VRAM. WSL VRAM reporting has
been buggy (ROCm/ROCm#6022, fixed around ROCDXG 1.2.1); if large batches OOM earlier than the
reported size suggests, raise `memory=` in `C:\Users\<you>\.wslconfig`.

**Performance ceiling.** Community measurements put WSL at roughly 70–80% of native Linux on
this hardware; hipBLASLt kernel coverage for gfx1151 is also weaker than for CDNA parts
(PyTorch ≤ 2.10 does not even list gfx1151 as hipBLASLt-supported). For serious throughput,
native Linux with `/dev/kfd` is the better target — `ARCH=rocm` in the same scripts covers it.
