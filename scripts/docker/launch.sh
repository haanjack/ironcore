#!/bin/bash

source .env

# Default to cuda if not set
VENDOR=${VENDOR:-"cuda"}

if [ "$VENDOR" == "rocm" ]; then
    GPU_FLAGS="--device /dev/kfd --device /dev/dri --security-opt seccomp=unconfined --group-add video"
    IMAGE="ironcore:rocm"
else
    GPU_FLAGS="--gpus=all"
    IMAGE="ironcore:cuda"
fi

echo "Launching $VENDOR container..."

docker run --rm -ti -u $(id -u):$(id -g) \
    --name=ironcore \
    $GPU_FLAGS --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
    -e HOME=/workspace \
    -p 6006:6006 \
    -v $(pwd):/workspace \
    -v /etc/passwd:/etc/passwd:ro \
    -v $DATASET_DIR:$DATASET_DIR \
    -v $MODEL_DIR:$MODEL_DIR \
    $IMAGE
