"""Firedrake worker for Kubernetes Jobs (spec section 17).

CLI:
    python3 worker.py --job-id ... --x0 ... --y0 ... --sigma ... --alpha ...
                      --callback-host weakheat-api-svc

Mesh resolution, dt, t_end and solver settings are FIXED inside this file;
the public API can only vary the four validated physical parameters.

Result is POSTed to the internal API callback as JSON:
    {"job_id": ..., "status": "done", "times": [...], "frames": [...],
     "runtime_ms": ..., "shape": [33, 33]}
Frames are already in canonical 33x33 row-major (y, x) order.
On failure a short safe error message is sent -- never a traceback.
"""
import argparse
import json
import os
import sys
import time
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from solve_case import run_simulation, CaseParams, MESH_N  # noqa: E402

CALLBACK_PORT = int(os.environ.get("CALLBACK_PORT", "80"))
CALLBACK_PATH = "/internal/result/{job_id}"
CALLBACK_TIMEOUT_S = 30


def post(url, payload, token):
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        url, data=data, method="POST",
        headers={"Content-Type": "application/json",
                 "X-Callback-Token": token})
    with urllib.request.urlopen(req, timeout=CALLBACK_TIMEOUT_S) as r:
        return r.status


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--x0", type=float, required=True)
    parser.add_argument("--y0", type=float, required=True)
    parser.add_argument("--sigma", type=float, required=True)
    parser.add_argument("--alpha", type=float, required=True)
    parser.add_argument("--callback-host", required=True)
    args = parser.parse_args()

    callback_token = os.environ.get("CALLBACK_TOKEN", "")
    url = f"http://{args.callback_host}:{CALLBACK_PORT}" + \
        CALLBACK_PATH.format(job_id=args.job_id)

    try:
        p = CaseParams(args.x0, args.y0, args.sigma, args.alpha)
        p.validate()
        t0 = time.perf_counter()
        _, sols, runtime_ms = run_simulation(p)
        order = None
        order_path = os.path.join(os.path.dirname(__file__),
                                  "..", "data", "operators", "order.npy")
        # Jobs bake the canonical order in at image build time:
        order = np_load_order()
        frames = [row.tolist() for row in sols[:, order]]  # each 33*33 row-major
        payload = {
            "job_id": args.job_id,
            "status": "done",
            "times": (np_times()).tolist(),
            "frames": frames,
            "runtime_ms": runtime_ms,
            "total_wall_ms": (time.perf_counter() - t0) * 1000.0,
            "shape": [MESH_N + 1, MESH_N + 1],
        }
    except Exception as e:  # noqa: BLE001 -- deliberately broad
        payload = {"job_id": args.job_id, "status": "error",
                   "error": f"{type(e).__name__}: {e}"[:300]}

    status = post(url, payload, callback_token)
    print(f"callback POST {status} (job {args.job_id})")


def np_load_order():
    import numpy as np
    for cand in ("/opt/weakheat/order.npy",
                 os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              "order.npy")):
        if os.path.exists(cand):
            return np.load(cand)
    raise FileNotFoundError("canonical order.npy not found in image")


def np_times():
    from solve_case import TIMES
    import numpy as np
    return np.asarray(TIMES, dtype=float)


if __name__ == "__main__":
    main()
