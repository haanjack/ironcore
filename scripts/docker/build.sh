#!/bin/bash

set -a # automatically export all variables
if [ -f .env ]; then
    source .env
fi
set +a

# Default to cuda if not set
ARCH=${ARCH:-"cuda"}
github_access_token=${github_access_token:-""}

if [ "$ARCH" == "rocm" ]; then
    BASE_IMAGE="rocm/pytorch:rocm7.2_ubuntu24.04_py3.12_pytorch_release_2.8.0"
    TAG="ironcore:rocm"
else
    NGC_VERSION=${NGC_VERSION:-"25.12"}
    BASE_IMAGE="nvcr.io/nvidia/pytorch:${NGC_VERSION}-py3"
    TAG="ironcore:cuda"
fi

echo "Building for $ARCH using $BASE_IMAGE..."

docker build . -t "$TAG" \
    --build-arg BASE_IMAGE="$BASE_IMAGE"
