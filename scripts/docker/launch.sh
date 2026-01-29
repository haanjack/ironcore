#!/bin/bash

source .env

docker run --rm -ti -u $(id -u):$(id -g) \
    --name=ironcore \
    --gpus=all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
    -e HOME=/workspace \
    -p 6006:6006 \
    -v $(pwd):/workspace \
    -v /etc/passwd:/etc/passwd:ro \
    -v $DATASET_DIR:$DATASET_DIR \
    -v $MODEL_DIR:$MODEL_DIR \
    ironcore:dev
