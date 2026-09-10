"""Training of the weak-form neural surrogate.

Objective (spec sections 12-13):

    L = L_data + lambda_weak * L_weak + lambda_IC * L_IC

Phase 1 (warm-up): L = L_data + L_IC.
Phase 2: measure the magnitudes of the terms, pick lambda_weak so that
the weak term contributes the target fraction (default 20%), continue.

L_weak = mean( (M du/dt + alpha K u / m)^2 ) -- no neural Laplacian.
"""
import argparse
import copy
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
try:
    from .dataset import HeatDataset, evaluate_fields  # noqa: F401
    from .model import CoordinateMLP, make_features, gaussian, save_meta
    from .weak_loss import WeakResidual
except ImportError:
    from dataset import HeatDataset, evaluate_fields  # noqa: F401
    from model import CoordinateMLP, make_features, gaussian, save_meta
    from weak_loss import WeakResidual

CKPT_DIR = os.path.join(os.path.dirname(__file__), "..", "checkpoints")


def forward_batch(model, coords, params_b, t_vec, need_grad=True):
    """Evaluate u_theta and du_theta/dt for a batch of complete fields.

    coords [N,2], params_b [B,4], t_vec [B] -> u [B,N], du_dt [B,N].
    du_dt is node-wise: t is given per point, so autograd returns the
    derivative of each node value separately.
    """
    B, N = params_b.shape[0], coords.shape[0]
    ctx = torch.enable_grad() if need_grad else torch.no_grad()
    with ctx:
        x = coords[:, 0].view(1, N).expand(B, N)
        y = coords[:, 1].view(1, N).expand(B, N)
        x0 = params_b[:, 0].view(B, 1).expand(B, N)
        y0 = params_b[:, 1].view(B, 1).expand(B, N)
        sg = params_b[:, 2].view(B, 1).expand(B, N)
        al = params_b[:, 3].view(B, 1).expand(B, N)
        t_pts = t_vec.repeat_interleave(N).requires_grad_(True)  # [B*N]
        t = t_pts.view(B, N)
        feat = make_features(x, y, x0, y0, t, sg, al).reshape(-1, 7)
        u_flat = model(feat)
        u = u_flat.view(B, N)
        du_dt = None
        if need_grad:
            du_dt = torch.autograd.grad(u_flat.sum(), t_pts, create_graph=True)[0]
            du_dt = du_dt.view(B, N)
    return u, du_dt


def ic_loss(model, coords, params_b):
    """L_IC = || u_theta(x, 0) - u_0(x) ||^2 against the exact Gaussian."""
    B, N = params_b.shape[0], coords.shape[0]
    zeros = torch.zeros(B, dtype=torch.float32)
    u0, _ = forward_batch(model, coords, params_b, zeros, need_grad=False)
    target = gaussian(
        coords.view(1, N, 2),
        params_b[:, 0].view(B, 1), params_b[:, 1].view(B, 1),
        params_b[:, 2].view(B, 1))
    return torch.mean((u0 - target) ** 2)


def measure_losses(model, data, wr, generator, n_batches, batch_size, device):
    """Mean magnitudes of L_data / L_weak over a few batches (no update)."""
    model.eval()
    sums = {"data": 0.0, "weak": 0.0, "ic": 0.0}
    for _ in range(n_batches):
        ci, ti = data.sample_pairs("train", batch_size, generator)
        params_b = data.params[ci].to(device)
        t_vec = data.times[ti].to(device)
        u_fd = data.solutions[ci, ti].to(device)
        alpha = params_b[:, 3]
        u, du_dt = forward_batch(model, data.coords, params_b, t_vec)
        with torch.no_grad():
            sums["data"] += torch.mean((u - u_fd) ** 2).item()
            sums["weak"] += wr.weak_loss(u.detach(), du_dt.detach(), alpha).item()
            sums["ic"] += ic_loss(model, data.coords, params_b).item()
    model.train()
    return {k: v / n_batches for k, v in sums.items()}


@torch.no_grad()
def validate(model, data, wr, device, batch_cases=10):
    """Mean relative FE L2 error over the whole validation split."""
    cases = list(data.split("val"))
    rel = []
    for i in range(0, len(cases), batch_cases):
        chunk = cases[i:i + batch_cases]
        params_b = data.params[chunk].to(device)
        u_pred = evaluate_fields(model, data.coords.to(device), params_b, data.times.to(device))
        u_true = data.solutions[chunk].to(device)
        for b in range(len(chunk)):
            for k in range(len(data.times)):
                rel.append(wr.relative_l2(
                    u_pred[b, k:k + 1], u_true[b, k:k + 1]).item())
    return float(np.mean(rel))


def train(args):
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    data = HeatDataset(args.data)
    wr = WeakResidual(args.operators)

    model = CoordinateMLP().to(device)
    print(f"model parameters: {model.num_parameters()}")
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="min", factor=0.5, patience=10)
    generator = torch.Generator().manual_seed(args.seed + 1)

    steps_per_epoch = (data.n_train * (len(data.times) - 1)) // args.batch_size
    best_val, best_state, bad_epochs = float("inf"), None, 0
    lambda_weak = 0.0
    history = []

    for epoch in range(1, args.epochs + 1):
        t0 = time.perf_counter()
        ep = {"data": 0.0, "weak": 0.0, "ic": 0.0}
        for _ in range(steps_per_epoch):
            ci, ti = data.sample_pairs("train", args.batch_size, generator)
            params_b = data.params[ci].to(device)
            t_vec = data.times[ti].to(device)
            u_fd = data.solutions[ci, ti].to(device)

            u, du_dt = forward_batch(model, data.coords, params_b, t_vec)
            l_data = torch.mean((u - u_fd) ** 2)
            loss = l_data

            if not args.data_only and epoch > args.warmup_epochs:
                l_weak = wr.weak_loss(u, du_dt, params_b[:, 3])
                loss = loss + lambda_weak * l_weak
                ep["weak"] += l_weak.item()

            l_ic = ic_loss(model, data.coords, params_b)
            loss = loss + l_ic

            opt.zero_grad()
            loss.backward()
            opt.step()

            ep["data"] += l_data.item()
            ep["ic"] += l_ic.item()

        for k in ep:
            ep[k] /= steps_per_epoch
        ep["epoch"] = epoch
        ep["lambda_weak"] = lambda_weak
        ep["val_rel_l2"] = validate(model, data, wr, device)
        ep["sec"] = time.perf_counter() - t0
        history.append(ep)
        sched.step(ep["val_rel_l2"])

        if not args.data_only and epoch == args.warmup_epochs:
            m = measure_losses(model, data, wr, generator,
                               n_batches=25, batch_size=args.batch_size, device=device)
            lambda_weak = args.weak_target_frac * m["data"] / max(m["weak"], 1e-12)
            print(f"[lambda_weak] data={m['data']:.3e} weak={m['weak']:.3e} "
                  f"ic={m['ic']:.3e} -> lambda_weak={lambda_weak:.3e}")

        marker = ""
        if ep["val_rel_l2"] < best_val - 1e-5:
            best_val = ep["val_rel_l2"]
            best_state = copy.deepcopy(model.state_dict())
            torch.save({
                "state_dict": best_state,
                "in_dim": model.net[0].in_features,
                "hidden": model.net[0].out_features,
                "depth": sum(1 for m in model.net if isinstance(m, torch.nn.Linear)) - 1,
                "val_rel_l2": best_val,
                "lambda_weak": lambda_weak,
                "data_only": args.data_only,
            }, args.out)
            marker = " *"
        elif best_state is not None:
            bad_epochs += 1

        print(f"[{epoch:3d}/{args.epochs}] "
              f"data={ep['data']:.3e} weak={ep['weak']:.3e} ic={ep['ic']:.3e} "
              f"val_relL2={ep['val_rel_l2']:.4f} lr={opt.param_groups[0]['lr']:.1e} "
              f"({ep['sec']:.1f}s){marker}")

        if bad_epochs >= args.patience:
            print(f"early stopping (no improvement for {args.patience} epochs)")
            break

    save_meta(os.path.join(CKPT_DIR, "model_meta.json"), model,
              extra={"best_val_rel_l2": best_val,
                     "lambda_weak": lambda_weak,
                     "data_only": args.data_only})
    with open(os.path.splitext(args.out)[0] + "_history.json", "w") as f:
        json.dump(history, f, indent=2)
    print(f"best validation relative FE L2: {best_val:.4f}")
    print(f"checkpoint -> {args.out}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default=os.path.join(
        os.path.dirname(__file__), "..", "data", "raw", "heat_dataset.npz"))
    p.add_argument("--operators", default=os.path.join(
        os.path.dirname(__file__), "..", "data", "operators"))
    p.add_argument("--out", default=os.path.join(CKPT_DIR, "weakheat_best.pt"))
    p.add_argument("--epochs", type=int, default=150)
    p.add_argument("--warmup-epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--patience", type=int, default=25)
    p.add_argument("--weak-target-frac", type=float, default=0.2)
    p.add_argument("--data-only", action="store_true",
                   help="train the supervised-only baseline")
    p.add_argument("--device", default="cpu")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    train(args)


if __name__ == "__main__":
    main()
