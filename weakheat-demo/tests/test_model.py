"""Autograd time-derivative and model tests."""
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "ml"))
from model import CoordinateMLP, make_features, gaussian  # noqa: E402


def test_torch_time_derivative():
    """u(x, t) = x^2 t^3  ->  u_t = 3 x^2 t^2 (node-wise via sum trick)."""
    x = torch.rand(100)
    t = torch.rand(100, requires_grad=True)
    u = x ** 2 * t ** 3
    du_dt = torch.autograd.grad(u.sum(), t, create_graph=True)[0]
    assert torch.allclose(du_dt, 3 * x ** 2 * t ** 2, atol=1e-6)


def test_torch_time_derivative_through_mlp():
    """The same node-wise derivative must hold through the MLP."""
    model = CoordinateMLP()
    coords = torch.rand(50, 2)
    t_vec = torch.rand(4)
    N, B = 50, 4
    t_pts = t_vec.repeat_interleave(N).requires_grad_(True)
    feat = torch.stack([
        coords[:, 0].repeat(B), coords[:, 1].repeat(B),
        torch.zeros(B * N), torch.zeros(B * N), t_pts,
        torch.full((B * N,), 0.07), torch.full((B * N,), 0.01)], dim=-1)
    u = model(feat)
    du_dt = torch.autograd.grad(u.sum(), t_pts, create_graph=True)[0]
    assert du_dt.shape == u.shape
    # finite difference cross-check on one point
    i = 17
    h = 1e-4
    with torch.no_grad():
        fp = model(feat.clone())
    feat_p, feat_m = feat.clone(), feat.clone()
    feat_p[i, 4] += h
    feat_m[i, 4] -= h
    with torch.no_grad():
        fd = (model(feat_p)[i] - model(feat_m)[i]) / (2 * h)
    assert abs(du_dt[i].item() - fd.item()) < 1e-2


def test_model_output_shape():
    model = CoordinateMLP()
    out = model(torch.rand(13, 7))
    assert out.shape == (13,)
    out = model(torch.rand(4, 5, 7))
    assert out.shape == (4, 5)


def test_model_parameter_count():
    model = CoordinateMLP()
    n = model.num_parameters()
    # 7-128-128-128-128-1 with biases
    expected = (7 * 128 + 128) + 3 * (128 * 128 + 128) + (128 + 1)
    assert n == expected


def test_gaussian_exact():
    coords = torch.tensor([[0.5, 0.5], [0.0, 0.0]])
    g = gaussian(coords, 0.5, 0.5, 0.1)
    assert abs(g[0].item() - 1.0) < 1e-6
    assert abs(g[1].item() - np.exp(-0.25 / 0.02)) < 1e-5


import numpy as np  # noqa: E402
