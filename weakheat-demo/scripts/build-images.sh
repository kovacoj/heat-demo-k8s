#!/usr/bin/env bash
# Build both images (run from the weakheat-demo root).
set -euo pipefail
cd "$(dirname "$0")/.."

API_TAG=${API_TAG:-cerit.io/kovacoj1/weakheat-api:dev}
FD_TAG=${FD_TAG:-cerit.io/kovacoj1/weakheat-firedrake:dev}
BASE_IMAGE=${BASE_IMAGE:-cerit.io/kovacoj1/heat-firedrake:20260909-081208}

docker build -t "$API_TAG" -f api/Dockerfile .
docker build --build-arg BASE_IMAGE="$BASE_IMAGE" -t "$FD_TAG" -f firedrake/Dockerfile .
echo "built:"
echo "  $API_TAG"
echo "  $FD_TAG"
