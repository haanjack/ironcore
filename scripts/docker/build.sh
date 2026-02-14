#!/bin/bash

set -a # automatically export all variables
if [ -f .env ]; then
    source .env
fi
set +a

# Default to cuda if not set
VENDOR=${VENDOR:-"cuda"}
github_access_token=${github_access_token:-""}

if [ "$VENDOR" == "rocm" ]; then
    BASE_IMAGE="rocm/pytorch:rocm6.2.4_ubuntu22.04_py3.10_pytorch_release_2.3.0"
    TAG="ironcore:rocm"
else
    NGC_VERSION=${NGC_VERSION:-"25.09"}
    BASE_IMAGE="nvcr.io/nvidia/pytorch:${NGC_VERSION}-py3"
    TAG="ironcore:cuda"
fi

echo "Building for $VENDOR using $BASE_IMAGE..."

docker build . -t $TAG \
    --build-arg BASE_IMAGE=$BASE_IMAGE \
    --build-arg ACCESS_TOKEN=$github_access_token
