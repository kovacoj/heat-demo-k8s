#!/usr/bin/env bash
# Apply all manifests in kovacovsky-ns (assumes kubectl is logged in).
set -euo pipefail
cd "$(dirname "$0")/.."

NS=${NS:-kovacovsky-ns}

kubectl apply -f k8s/rbac.yaml -n "$NS"
kubectl apply -f k8s/deployment.yaml -n "$NS"
kubectl apply -f k8s/service.yaml -n "$NS"
kubectl apply -f k8s/ingress.yaml -n "$NS"

kubectl rollout status deployment/weakheat-api -n "$NS" --timeout=120s
echo
kubectl get pods,svc,ingress -n "$NS" | grep weakheat || true
