"""Evaluation on held-out test cases + presentation artifacts.

Reports (never fabricated -- everything measured):
    median / p90 / max relative FE L2 error over (case, time)
    mass conservation drift of the neural solution
    mean scaled weak residual
    NN inference runtime for all 26 frames vs Firedrake solve runtime

Artifacts:
    presentation/metrics.json
    presentation/example_comparison.png   (Firedrake | NN | abs error)
    presentation/runtime_comparison.pdf
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
    from .weak_loss import WeakResidual
except ImportError:
    from dataset import HeatDataset, evaluate_fields
    from model import load_model
    from weak_loss import WeakResidual

ROOT = os.path.join(os.path.dirname(__file__), "..")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default=os.path.join(ROOT, "checkpoints", "weakheat_best.pt"))
    p.add_argument("--data-only-checkpoint", default=None)
    p.add_argument("--data", default=os.path.join(ROOT, "data", "raw", "heat_dataset.npz"))
    p.add_argument("--operators", default=os.path.join(ROOT, "data", "operators"))
    p.add_argument("--out-dir", default=os.path.join(ROOT, "presentation"))
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    data = HeatDataset(args.data)
    wr = WeakResidual(args.operators)
    model = load_model(args.checkpoint)
    model.eval()
    print(f"model parameters: {model.num_parameters()}")

    test_cases = list(data.split("test"))
    params = data.params[test_cases]
    u_true = data.solutions[test_cases]
    u_pred = evaluate_fields(model, data.coords, params, data.times)

    # --- relative FE L2 over every (case, time) -------------------------
    rel = wr.relative_l2(
        u_pred.reshape(-1, u_pred.shape[-1]),
        u_true.reshape(-1, u_true.shape[-1])).numpy()
    rel_per_case_max = rel.reshape(len(test_cases), len(data.times)).max(axis=1)

    # --- mass conservation of the neural solution ----------------------
    mass0 = wr.total_mass(u_pred[:, 0, :])
    mass_drift = (wr.total_mass(u_pred[:, -1, :]) - mass0) / mass0.clamp_min(1e-12)

    # --- weak residual (sampled over all test cases/times) -------------
    from train import forward_batch
    with torch.enable_grad():
        B, T, N = u_pred.shape
        resid_sq = []
        for i in range(0, B, 10):
            chunk = torch.arange(i, min(i + 10, B))
            for k in range(T):
                t_vec = data.times[k].repeat(len(chunk))
                u, du_dt = forward_batch(model, data.coords, params[chunk], t_vec)
                r = wr.scaled_residual(u, du_dt, params[chunk, 3])
                resid_sq.append(torch.mean(r ** 2).item())
        weak_resid_mean = float(np.sqrt(np.mean(resid_sq)))

    metrics = {
        "pde": "2D heat equation (Neumann)",
        "mesh": "32x32 CG1",
        "ndofs": int(u_pred.shape[-1]),
        "model_parameters": int(model.num_parameters()),
        "n_test_cases": len(test_cases),
        "test_relative_l2_median": float(np.median(rel)),
        "test_relative_l2_p90": float(np.percentile(rel, 90)),
        "test_relative_l2_max": float(rel.max()),
        "test_relative_l2_worst_case_max": float(rel_per_case_max.max()),
        "mass_drift_median": float(np.median(mass_drift.numpy())),
        "mass_drift_max": float(np.abs(mass_drift.numpy()).max()),
        "weak_residual_mean": weak_resid_mean,
        "firedrake_runtime_ms_median": float(np.median(data.solver_runtime_ms)),
    }

    # --- NN inference runtime: all 26 frames, median of repeats --------
    with torch.no_grad():
        for _ in range(5):  # warm-up
            evaluate_fields(model, data.coords, params[:1], data.times)
        timings = []
        for _ in range(20):
            t0 = time.perf_counter()
            evaluate_fields(model, data.coords, params[:1], data.times)
            timings.append((time.perf_counter() - t0) * 1000.0)
    metrics["nn_inference_ms_median_26_frames"] = float(np.median(timings))
    metrics["speedup_vs_firedrake"] = (
        metrics["firedrake_runtime_ms_median"] / metrics["nn_inference_ms_median_26_frames"])

    # optional supervised-only baseline
    if args.data_only_checkpoint and os.path.exists(args.data_only_checkpoint):
        base = load_model(args.data_only_checkpoint)
        u_base = evaluate_fields(base, data.coords, params, data.times)
        rel_b = wr.relative_l2(
            u_base.reshape(-1, u_base.shape[-1]),
            u_true.reshape(-1, u_true.shape[-1])).numpy()
        metrics["data_only_relative_l2_median"] = float(np.median(rel_b))

    with open(os.path.join(args.out_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    for k, v in metrics.items():
        print(f"{k:38s} {v}")

    # --- example comparison figure --------------------------------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ci = 0
    ti = len(data.times) // 2
    order = data.order
    gs = (33, 33)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
    fd = u_true[ci, ti].numpy()[order].reshape(gs)
    nn = u_pred[ci, ti].detach().numpy()[order].reshape(gs)
    err = np.abs(fd - nn)
    for ax, img, title, cmap in (
            (axes[0], fd, f"Firedrake FEM  t={float(data.times[ti]):.2f}", "inferno"),
            (axes[1], nn, "Weak neural surrogate", "inferno"),
            (axes[2], err, "|FEM - NN|", "viridis")):
        im = ax.imshow(img, origin="lower", cmap=cmap,
                       vmin=0 if cmap == "inferno" else None,
                       vmax=1 if cmap == "inferno" else None)
        ax.set_title(title)
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out_dir, "example_comparison.png"), dpi=140)
    print(f"figure -> {args.out_dir}/example_comparison.png")

    # --- runtime comparison figure --------------------------------------
    fig, ax = plt.subplots(figsize=(5.5, 3.4))
    bars = ["Firedrake\n(transient solve)", "NN\n(26 frames, 1 pass)"]
    vals = [metrics["firedrake_runtime_ms_median"],
            metrics["nn_inference_ms_median_26_frames"]]
    ax.bar(bars, vals, color=["#4477aa", "#ee6677"])
    for i, v in enumerate(vals):
        ax.text(i, v * 1.05, f"{v:.1f} ms", ha="center")
    ax.set_yscale("log")
    ax.set_ylabel("wall-clock [ms]")
    ax.set_title(f"speedup ~{metrics['speedup_vs_firedrake']:.0f}x")
    fig.tight_layout()
    fig.savefig(os.path.join(args.out_dir, "runtime_comparison.pdf"))
    print(f"figure -> {args.out_dir}/runtime_comparison.pdf")


if __name__ == "__main__":
    main()
