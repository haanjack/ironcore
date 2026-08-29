#!/bin/bash

# Load .env if present (non-fatal — defaults are set below)
[[ -f .env ]] && source .env

if [ $# -eq 0 ]; then
    COMMAND=(bash)
else
    COMMAND=("$@")
fi

# Defaults for optional mount paths
DATASET_DIR=${DATASET_DIR:-""}
MODEL_DIR=${MODEL_DIR:-""}

# Auto-detect GPU architecture from device files unless ARCH is explicitly set
if [ -z "$ARCH" ]; then
    if [ -e /dev/nvidiactl ]; then
        ARCH="cuda"
    elif [ -e /dev/dxg ]; then
        ARCH="rocm-wsl"
    elif [ -e /dev/kfd ]; then
        ARCH="rocm"
    else
        ARCH="cuda"
    fi
fi

case "$ARCH" in
    rocm-wsl)
        # librocdxg is baked into the image (see Dockerfile ROCDXG_VERSION), so the
        # WSL distro needs no ROCm install — only libdxcore.so comes from the host,
        # since it is supplied by the Windows driver and cannot be shipped in a layer.
        #
        # HSA_ENABLE_DXG_DETECTION is required for ROCm < 7.13; harmless on newer.
        # TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL: PyTorch <= 2.12 ships an AOTriton
        # where gfx1151 is still flagged experimental, and without this SDPA silently
        # falls back to the math backend (roughly an order of magnitude slower).
        GPU_FLAGS="-v /usr/lib/wsl/lib/libdxcore.so:/usr/lib/libdxcore.so \
            --device=/dev/dxg \
            --cap-add=SYS_PTRACE \
            -e HSA_ENABLE_DXG_DETECTION=1 \
            -e TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1 \
            --security-opt seccomp=unconfined \
            --shm-size 8G"
        IMAGE="ironcore:rocm-wsl"
        ;;
    rocm)
        GPU_FLAGS="--device /dev/kfd --device /dev/dri --security-opt seccomp=unconfined --group-add video"
        IMAGE="ironcore:rocm"
        ;;
    *)
        GPU_FLAGS="--gpus=all"
        IMAGE="ironcore:cuda"
        ;;
esac

# Build optional volume mounts (skip when empty)
VOLUME_MOUNTS=""
[[ -n "$DATASET_DIR" ]] && VOLUME_MOUNTS="$VOLUME_MOUNTS -v $DATASET_DIR:$DATASET_DIR"
[[ -n "$MODEL_DIR" ]] && VOLUME_MOUNTS="$VOLUME_MOUNTS -v $MODEL_DIR:$MODEL_DIR"

echo "Launching $ARCH container ($IMAGE)..."

# -ti when stdin is a terminal (interactive), -t otherwise
TTY_FLAGS="-t"
[ -t 0 ] && TTY_FLAGS="-ti"

# HF_HUB_DISABLE_XET: the xet transfer backend 404s against the Hub API from inside the
# container, which fails every hf_hub-marked test and any tokenizer/weight download.
# The classic HTTP path works, so turn xet off.
exec docker run --rm $TTY_FLAGS -u $(id -u):$(id -g) \
    --name=ironcore \
    $GPU_FLAGS --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
    -e HOME=/workspace \
    -e HF_HUB_DISABLE_XET=1 \
    -w /workspace \
    -p 6006:6006 \
    -v $(pwd):/workspace \
    -v /etc/passwd:/etc/passwd:ro \
    $VOLUME_MOUNTS \
    $IMAGE "${COMMAND[@]}"
