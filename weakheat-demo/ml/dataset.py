"""Loading of the Firedrake ground-truth dataset and FE operators."""
import os

import numpy as np
import scipy.sparse as sp
import torch

try:
    from .model import make_features  # package-style (api image)
except ImportError:  # script-style (python3 ml/train.py)
    from model import make_features

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
DATASET_PATH = os.path.join(DATA_DIR, "raw", "heat_dataset.npz")
OPERATORS_DIR = os.path.join(DATA_DIR, "operators")


class HeatDataset:
    """Holds params, times, solutions and provides (case, time) pair sampling.

    solutions[i, k, :] is the Firedrake solution of case i at times[k] in
    the native Firedrake dof order (identical for every case).
    """

    def __init__(self, path=DATASET_PATH):
        d = np.load(path)
        self.params = torch.tensor(d["params"], dtype=torch.float32)      # [C,4]
        self.times = torch.tensor(d["times"], dtype=torch.float32)        # [T]
        self.solutions = torch.tensor(d["solutions"], dtype=torch.float32)  # [C,T,N]
        self.coords = torch.tensor(d["coords"], dtype=torch.float32)      # [N,2]
        self.solver_runtime_ms = d["solver_runtime_ms"]
        self.n_train, self.n_val, self.n_test = int(d["n_train"]), int(d["n_val"]), int(d["n_test"])
        self.order = np.load(os.path.join(OPERATORS_DIR, "order.npy"))

    def split(self, name):
        if name == "train":
            return range(0, self.n_train)
        if name == "val":
            return range(self.n_train, self.n_train + self.n_val)
        if name == "test":
            return range(self.n_train + self.n_val, self.solutions.shape[0])
        raise ValueError(name)

    def sample_pairs(self, split, batch_size, generator):
        """Random (case, time) pairs, excluding t=0 (handled by the IC loss)."""
        cases = torch.tensor(list(self.split(split)))
        n_times = len(self.times) - 1
        ci = cases[torch.randint(len(cases), (batch_size,), generator=generator)]
        ti = 1 + torch.randint(n_times, (batch_size,), generator=generator)
        return ci, ti

    def sample_cases(self, split, batch_size, generator):
        cases = torch.tensor(list(self.split(split)))
        return cases[torch.randint(len(cases), (batch_size,), generator=generator)]


def load_operators(operators_dir=OPERATORS_DIR):
    """M, K as torch sparse COO tensors + lumped mass vector."""
    out = []
    for name in ("M", "K"):
        A = sp.load_npz(os.path.join(operators_dir, f"{name}.npz")).tocoo()
        idx = np.stack([A.row, A.col]).astype(np.int64)
        val = torch.tensor(A.data, dtype=torch.float32)
        out.append(torch.sparse_coo_tensor(idx, val, A.shape).coalesce())
    M, K = out
    lumped = torch.sparse.mm(M, torch.ones((M.shape[0], 1))).squeeze(1)
    return M, K, lumped


def evaluate_fields(model, coords, params, times):
    """Evaluate the network on the full FE grid.

    coords [N,2], params [B,4], times [T] -> u [B, T, N] (no grad tracking).
    """
    model.eval()
    B, N, T = params.shape[0], coords.shape[0], times.shape[0]
    with torch.no_grad():
        x = coords[:, 0].view(1, 1, N).expand(B, T, N)
        y = coords[:, 1].view(1, 1, N).expand(B, T, N)
        x0 = params[:, 0].view(B, 1, 1).expand(B, T, N)
        y0 = params[:, 1].view(B, 1, 1).expand(B, T, N)
        sg = params[:, 2].view(B, 1, 1).expand(B, T, N)
        al = params[:, 3].view(B, 1, 1).expand(B, T, N)
        t = times.view(1, T, 1).expand(B, T, N)
        feat = make_features(x, y, x0, y0, t, sg, al)
        u = model(feat.reshape(-1, 7)).reshape(B, T, N)
    return u
