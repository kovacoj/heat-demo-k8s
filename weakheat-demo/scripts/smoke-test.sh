#!/usr/bin/env bash
# Smoke test the deployed API (spec section 32).
#   ./scripts/smoke-test.sh https://weakheat-kovacovsky-ns.dyn.cloud.e-infra.cz TOKEN
set -euo pipefail

HOST=${1:?usage: smoke-test.sh https://host token}
TOKEN=${2:?missing token}

echo "== health"
curl -fsS "$HOST/api/health"; echo

echo "== neural inference"
NN_JSON=$(curl -fsS -X POST "$HOST/api/nn/predict" \
    -H "Authorization: Bearer $TOKEN" -H "Content-Type: application/json" \
    -d '{"x0":0.4,"y0":0.6,"sigma":0.07,"alpha":0.01}')
echo "$NN_JSON" | python3 -c '
import json, sys, math
r = json.load(sys.stdin)
assert len(r["times"]) == 26, "expected 26 times"
assert len(r["frames"]) == 26 and len(r["frames"][0]) == 33 * 33, "bad frame shape"
assert all(math.isfinite(v) for fr in r["frames"] for v in fr), "non-finite values"
assert math.isfinite(r["inference_ms"]) and r["inference_ms"] >= 0
print("OK: 26 frames, %s, %.1f ms" % (r["shape"], r["inference_ms"]))
'

echo "== firedrake job"
RUN_JSON=$(curl -fsS -X POST "$HOST/api/firedrake/run" \
    -H "Authorization: Bearer $TOKEN" -H "Content-Type: application/json" \
    -d '{"x0":0.4,"y0":0.6,"sigma":0.07,"alpha":0.01}')
JOB_ID=$(echo "$RUN_JSON" | python3 -c 'import json,sys; print(json.load(sys.stdin)["job_id"])')
echo "job_id: $JOB_ID  (weakheat-fd-$JOB_ID)"

echo "== waiting for result"
for i in $(seq 1 240); do
    ST=$(curl -fsS "$HOST/api/firedrake/$JOB_ID" -H "Authorization: Bearer $TOKEN")
    STATUS=$(echo "$ST" | python3 -c 'import json,sys; print(json.load(sys.stdin)["status"])')
    echo "  [$i] $STATUS"
    if [ "$STATUS" = "done" ]; then
        echo "$ST" | python3 -c '
import json, sys
r = json.load(sys.stdin)
assert len(r["frames"]) == 26 and len(r["frames"][0]) == 33 * 33
print("OK: firedrake runtime %.0f ms, relative L2 %.2f %%"
      % (r["runtime_ms"], 100 * r.get("relative_l2", float("nan"))))
'
        exit 0
    fi
    if [ "$STATUS" = "error" ]; then
        echo "FAILED: $ST"; exit 1
    fi
    sleep 2
done
echo "timeout waiting for job"; exit 1
