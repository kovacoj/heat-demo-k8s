"""Operator sanity tests: shapes, symmetry, Neumann property K*1 ~ 0."""
import os
import sys

import numpy as np
import scipy.sparse as sp
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "ml"))
from dataset import load_operators  # noqa: E402

OPERATORS = os.path.join(os.path.dirname(__file__), "..", "data", "operators")


def test_operator_shapes():
    M, K, lumped = load_operators(OPERATORS)
    assert M.shape == (1089, 1089)
    assert K.shape == (1089, 1089)
    assert lumped.shape == (1089,)


def test_mass_matrix_symmetric():
    M = sp.load_npz(os.path.join(OPERATORS, "M.npz"))
    assert np.allclose(M.toarray(), M.toarray().T, atol=1e-12)


def test_stiffness_matrix_symmetric():
    K = sp.load_npz(os.path.join(OPERATORS, "K.npz"))
    assert np.allclose(K.toarray(), K.toarray().T, atol=1e-12)


def test_constant_field_Ku_approximately_zero():
    """Neumann boundary: K applied to a constant field vanishes."""
    K = sp.load_npz(os.path.join(OPERATORS, "K.npz"))
    Ku1 = K @ np.ones(K.shape[0])
    assert np.abs(Ku1).max() < 1e-10


def test_lumped_mass_total_is_domain_area():
    M, K, lumped = load_operators(OPERATORS)
    assert abs(float(lumped.sum()) - 1.0) < 1e-5  # float32 accumulation


def test_torch_sparse_matches_scipy():
    M, K, lumped = load_operators(OPERATORS)
    Ms = sp.load_npz(os.path.join(OPERATORS, "M.npz"))
    x = torch.linspace(0, 1, 1089)
    y = torch.sparse.mm(M, x.view(-1, 1)).squeeze(1)
    y_ref = torch.tensor(Ms @ x.numpy(), dtype=torch.float32)
    assert torch.allclose(y, y_ref, atol=1e-6)
