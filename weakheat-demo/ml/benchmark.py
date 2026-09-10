"""Standalone runtime benchmark: NN inference vs Firedrake transient solve.

Rules (spec section 16): benchmark compute only.  The model is loaded
before timing, several warm-up evaluations run, then the full 26-frame
request is timed; the median over repeats is reported.
"""
import argparse
import json
import os
import sys

import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
try:
    from .dataset import HeatDataset, evaluate_fields
    from .model import load_model
except ImportError:
    from dataset import HeatDataset, evaluate_fields
    from model import load_model

ROOT = os.path.join(os.path.dirname(__file__), "..")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default=os.path.join(ROOT, "checkpoints", "weakheat_best.pt"))
    p.add_argument("--data", default=os.path.join(ROOT, "data", "raw", "heat_dataset.npz"))
    p.add_argument("--repeats", type=int, default=50)
    args = p.parse_args()

    data = HeatDataset(args.data)
    model = load_model(args.checkpoint)
    model.eval()
    params = data.params[:1]
    times = data.times

    with torch.no_grad():
        for _ in range(10):
            evaluate_fields(model, data.coords, params, times)
        timings = []
        for _ in range(args.repeats):
            t0 = time.perf_counter()
            evaluate_fields(model, data.coords, params, times)
            timings.append((time.perf_counter() - t0) * 1000.0)

    out = {
        "nn_inference_ms_median_26_frames": float(np.median(timings)),
        "nn_inference_ms_min": float(np.min(timings)),
        "nn_inference_ms_max": float(np.max(timings)),
        "firedrake_runtime_ms_median": float(np.median(data.solver_runtime_ms)),
        "repeats": args.repeats,
    }
    out["speedup_vs_firedrake"] = (
        out["firedrake_runtime_ms_median"] / out["nn_inference_ms_median_26_frames"])
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
