"""Assemble and export the FE mass and stiffness matrices from Firedrake.

The weak form of the Neumann heat equation, tested against the complete
CG1 basis, is

    M dc/dt + alpha K c = 0,

with

    M_ij = int phi_i phi_j dx,     K_ij = int grad(phi_i) . grad(phi_j) dx.

These matrices are the bridge between Firedrake (FE test space) and the
PyTorch neural trial field: the network only ever supplies c_theta(t) and
its first time derivative.  No second spatial derivative of the network
is ever evaluated.

Outputs (data/operators/):
    M.npz, K.npz     scipy CSR matrices
    coords.npy       [Ndof, 2] node coordinates in native dof order
    order.npy        canonical plot ordering (lexsort by x, then y)
    meta.json        discretisation metadata
"""
import json
import os

import numpy as np
import scipy.sparse as sp
from firedrake import (
    UnitSquareMesh, FunctionSpace, Function, TrialFunction, TestFunction,
    inner, grad, dx, interpolate, SpatialCoordinate,
)

from solve_case import MESH_N, ELEMENT, DT, T_END, OUTPUT_DT, TIMES, PARAM_LIMITS

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "operators")


def petsc_to_csr(matrix):
    """Firedrake Matrix (aij) -> scipy CSR."""
    ai, aj, av = matrix.petscmat.getValuesCSR()
    n, m = matrix.petscmat.getSize()
    return sp.csr_matrix((np.asarray(av), np.asarray(aj), np.asarray(ai)),
                         shape=(n, m))


def main():
    mesh = UnitSquareMesh(MESH_N, MESH_N)
    V = FunctionSpace(mesh, *ELEMENT)
    ndofs = V.dof_count
    print(f"mesh {MESH_N}x{MESH_N} {ELEMENT[0]}{ELEMENT[1]}: {ndofs} dofs")

    trial, test = TrialFunction(V), TestFunction(V)
    M = petsc_to_csr(assemble_fd(inner(trial, test) * dx))
    K = petsc_to_csr(assemble_fd(inner(grad(trial), grad(test)) * dx))

    # node coordinates in native dof order (consistent with solution vectors)
    x, y = SpatialCoordinate(mesh)
    X = Function(V).interpolate(x)
    Y = Function(V).interpolate(y)
    coords = np.column_stack([
        X.dat.data_ro.copy(),
        Y.dat.data_ro.copy(),
    ])

    # canonical plotting order: primary key y, secondary key x
    order = np.lexsort((coords[:, 0], coords[:, 1]))

    # --- verification ---------------------------------------------------
    assert M.shape == (ndofs, ndofs) and K.shape == (ndofs, ndofs)
    assert np.allclose(M.toarray(), M.toarray().T, atol=1e-12), "M not symmetric"
    assert np.allclose(K.toarray(), K.toarray().T, atol=1e-12), "K not symmetric"
    # Neumann boundary: K applied to a constant field is ~0
    Ku1 = K @ np.ones(ndofs)
    print(f"symmetry OK; ||K 1||_inf = {np.abs(Ku1).max():.3e}")
    assert np.abs(Ku1).max() < 1e-10, "K*1 should vanish for Neumann BC"

    # orientation check: u = x + 10 y must reshape to rows=y, cols=x
    u_test = coords[:, 0] + 10.0 * coords[:, 1]
    grid = u_test[order].reshape(MESH_N + 1, MESH_N + 1)
    assert np.allclose(grid[0, :], np.linspace(0, 1, MESH_N + 1)), "row 0 must be y=0 varying in x"
    assert np.allclose(grid[:, 3], 3 / MESH_N + 10 * np.linspace(0, 1, MESH_N + 1)), "col must vary in y"
    print("canonical order verified: reshape(N+1, N+1) -> [y, x]")

    lumped = np.asarray(M.sum(axis=1)).ravel()
    print(f"lumped mass total (should be 1.0): {lumped.sum():.12f}")

    os.makedirs(OUT_DIR, exist_ok=True)
    sp.save_npz(os.path.join(OUT_DIR, "M.npz"), M)
    sp.save_npz(os.path.join(OUT_DIR, "K.npz"), K)
    np.save(os.path.join(OUT_DIR, "coords.npy"), coords)
    np.save(os.path.join(OUT_DIR, "order.npy"), order)
    meta = {
        "mesh": f"{MESH_N}x{MESH_N}",
        "element": f"{ELEMENT[0]}{ELEMENT[1]}",
        "ndofs": int(ndofs),
        "dt": DT,
        "t_end": T_END,
        "output_dt": OUTPUT_DT,
        "times": TIMES.tolist(),
        "param_limits": PARAM_LIMITS,
        "grid_shape": [MESH_N + 1, MESH_N + 1],
    }
    with open(os.path.join(OUT_DIR, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"saved operators -> {os.path.abspath(OUT_DIR)}")


def assemble_fd(form):
    """Assemble a bilinear form as an AIJ matrix."""
    from firedrake import assemble as fd_assemble
    return fd_assemble(form, mat_type="aij")


if __name__ == "__main__":
    main()
