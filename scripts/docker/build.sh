#!/bin/bash

set -a # automatically export all variables
if [ -f .env ]; then
    source .env
fi
set +a

# Default to empty token if not set
github_access_token=${github_access_token:-""}

docker build . -t ironcore:dev \
    --build-arg NGC_VERSION=25.09 \
    --build-arg ACCESS_TOKEN=$github_access_token
