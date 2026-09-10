"""Weak residual and dataset tests."""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "ml"))
from dataset import HeatDataset, load_operators  # noqa: E402
from weak_loss import WeakResidual  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..")


def test_weak_residual_of_constant_field_is_zero():
    """u = const, du/dt = 0  ->  r = alpha K const ~ 0 (Neumann)."""
    wr = WeakResidual(os.path.join(ROOT, "data", "operators"))
    u = torch.full((3, wr.N), 0.5)
    du = torch.zeros((3, wr.N))
    alpha = torch.tensor([0.02, 0.01, 0.005])
    r = wr.residual(u, du, alpha)
    assert torch.abs(r).max() < 1e-7


def test_weak_residual_exact_for_steady_harmonic_check():
    """r must vanish when M du/dt + alpha K u = 0 exactly: pick u = const."""
    wr = WeakResidual(os.path.join(ROOT, "data", "operators"))
    u = torch.rand(2, wr.N)
    # du/dt chosen as -alpha M^-1 K u is the exact semi-discrete solution
    M = wr.M.to_dense()
    K = wr.K.to_dense()
    alpha = torch.tensor([0.01, 0.02])
    Ku = u @ K.T                                   # [B,N]
    rhs = (-alpha.view(-1, 1) * Ku).T              # [N,B]
    du = torch.linalg.solve(M, rhs).T              # [B,N]
    r = wr.residual(u, du, alpha)
    assert torch.abs(r).max() < 1e-4


def test_dataset_shapes_and_conservation():
    data = HeatDataset(os.path.join(ROOT, "data", "raw", "heat_dataset.npz"))
    C, T, N = data.solutions.shape
    assert (data.n_train, data.n_val, data.n_test) == (160, 20, 20)
    assert C == 200 and T == 26 and N == 1089
    assert torch.isfinite(data.solutions).all()
    # peak amplitude must decay for every case (diffusion)
    peak = data.solutions.max(dim=2).values
    assert (peak[:, 1:] <= peak[:, :-1] + 1e-6).all()
    # total heat conserved by the Firedrake reference
    wr = WeakResidual(os.path.join(ROOT, "data", "operators"))
    m0 = wr.total_mass(data.solutions[:, 0, :])
    m1 = wr.total_mass(data.solutions[:, -1, :])
    drift = ((m1 - m0) / m0).abs().max()
    assert drift < 1e-4


def test_relative_l2_zero_for_identical_fields():
    wr = WeakResidual(os.path.join(ROOT, "data", "operators"))
    u = torch.rand(4, wr.N)
    rel = wr.relative_l2(u, u)
    assert (rel < 1e-6).all()
    rel2 = wr.relative_l2(2 * u, u)
    assert torch.allclose(rel2, torch.ones(4), atol=1e-4)
