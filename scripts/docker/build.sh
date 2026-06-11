#!/bin/bash

set -a # automatically export all variables
if [ -f .env ]; then
    source .env
fi
set +a

# Default to cuda if not set
ARCH=${ARCH:-"cuda"}
github_access_token=${github_access_token:-""}

if [ "$ARCH" == "rocm" ] || [ "$ARCH" == "rocm-wsl" ]; then
    # ROCm PyTorch image — full tag or let the default apply.
    # WSL2 hosts can override ROCM_IMAGE for a different PyTorch version.
    # Example: ROCM_IMAGE="rocm/pytorch:rocm7.2.3_ubuntu24.04_py3.12_pytorch_release_2.10.0"
    BASE_IMAGE=${ROCM_IMAGE:-"rocm/pytorch:rocm7.2_ubuntu24.04_py3.12_pytorch_release_2.8.0"}
    # WSL2 + DXG: detect by /dev/dxg presence (or explicit rocm-wsl arch)
    if [ "$ARCH" == "rocm-wsl" ] || [ -e /dev/dxg ]; then
        TAG="ironcore:rocm-wsl"
    else
        TAG="ironcore:rocm"
    fi
else
    NGC_VERSION=${NGC_VERSION:-"25.12"}
    BASE_IMAGE="nvcr.io/nvidia/pytorch:${NGC_VERSION}-py3"
    TAG="ironcore:cuda"
fi

echo "Building for $ARCH ($TAG) using $BASE_IMAGE..."

docker build . -t "$TAG" \
    --build-arg BASE_IMAGE="$BASE_IMAGE"
