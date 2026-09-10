#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

API_TAG=${API_TAG:-cerit.io/kovacoj1/weakheat-api:dev}
FD_TAG=${FD_TAG:-cerit.io/kovacoj1/weakheat-firedrake:dev}

docker push "$API_TAG"
docker push "$FD_TAG"
