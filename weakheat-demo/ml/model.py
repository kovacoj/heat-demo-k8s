"""Coordinate MLP surrogate u_theta(x, y, t, x0, y0, sigma, alpha).

The network represents the continuous field; for the weak-form loss its
nodal values are interpreted as the coefficient vector c_theta(t) of the
Firedrake CG1 basis.  Only first-order/time information of the network is
ever differentiated -- the network Laplacian is never evaluated.
"""
import json

import torch
from torch import nn

# raw feature vector layout (dx = x - x0, dy = y - y0)
FEATURES = ["x", "y", "dx", "dy", "t", "sigma", "alpha"]
IN_DIM = len(FEATURES)
HIDDEN = 128
DEPTH = 4  # hidden layers

# normalization constants, mapping every feature approximately to [-1, 1]
FEATURE_MEAN = [0.5, 0.5, 0.5, 0.5, 0.125, 0.07, 0.0125]
FEATURE_SCALE = [0.5, 0.5, 0.5, 0.5, 0.125, 0.03, 0.0075]


class CoordinateMLP(nn.Module):
    def __init__(self, in_dim=IN_DIM, hidden=HIDDEN, depth=DEPTH):
        super().__init__()
        layers = [nn.Linear(in_dim, hidden), nn.Tanh()]
        for _ in range(depth - 1):
            layers += [nn.Linear(hidden, hidden), nn.Tanh()]
        layers += [nn.Linear(hidden, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, inp):
        """inp: [..., 7] raw features -> [..., 1] field value."""
        return self.net(normalize_features(inp)).squeeze(-1)

    def num_parameters(self):
        return sum(p.numel() for p in self.parameters())


def make_features(x, y, x0, y0, t, sigma, alpha):
    """Raw feature tensor [..., 7]; relative coordinates improve conditioning."""
    return torch.stack([x, y, x - x0, y - y0, t, sigma, alpha], dim=-1)


def normalize_features(feat):
    mean = torch.tensor(FEATURE_MEAN, dtype=feat.dtype, device=feat.device)
    scale = torch.tensor(FEATURE_SCALE, dtype=feat.dtype, device=feat.device)
    return (feat - mean) / scale


def gaussian(coords, x0, y0, sigma):
    """Exact Gaussian initial condition at node coordinates."""
    r2 = (coords[..., 0] - x0) ** 2 + (coords[..., 1] - y0) ** 2
    return torch.exp(-r2 / (2 * sigma ** 2))


def meta_dict(model=None):
    return {
        "architecture": "CoordinateMLP",
        "in_dim": IN_DIM,
        "hidden": HIDDEN,
        "depth": DEPTH,
        "features": FEATURES,
        "feature_mean": FEATURE_MEAN,
        "feature_scale": FEATURE_SCALE,
        "activation": "tanh",
        "model_parameters": int(model.num_parameters()) if model is not None else None,
    }


def save_meta(path, model=None, extra=None):
    d = meta_dict(model)
    if extra:
        d.update(extra)
    with open(path, "w") as f:
        json.dump(d, f, indent=2)


def load_model(checkpoint_path, device="cpu"):
    ckpt = torch.load(checkpoint_path, map_location=device)
    model = CoordinateMLP(ckpt["in_dim"], ckpt["hidden"], ckpt["depth"])
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model
