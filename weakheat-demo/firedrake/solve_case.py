"""Transient 2D heat equation on the unit square with Firedrake.

    du/dt - alpha * laplace(u) = 0   in Omega = (0,1)^2
    grad(u) . n = 0                 on dOmega       (natural BC)
    u(x,y,0) = A * exp(-((x-x0)^2 + (y-y0)^2) / (2 sigma^2))

Time discretisation: implicit Euler.  No boundary condition is imposed;
the homogeneous Neumann condition arises naturally from the weak form.
"""
import json
import time
from dataclasses import dataclass, asdict

import numpy as np
from firedrake import (
    UnitSquareMesh, FunctionSpace, Function, TrialFunction, TestFunction,
    inner, grad, dx, SpatialCoordinate, exp,
    LinearVariationalProblem, LinearVariationalSolver,
)

# ---------------------------------------------------------------------------
# Fixed discretisation parameters (public API must not change these)
# ---------------------------------------------------------------------------
MESH_N = 32
ELEMENT = ("CG", 1)
DT = 0.005
T_END = 0.25
OUTPUT_DT = 0.01
TIMES = np.round(np.arange(0.0, T_END + 1e-12, OUTPUT_DT), 10)  # 26 values

PARAM_LIMITS = {
    "x0": (0.25, 0.75),
    "y0": (0.25, 0.75),
    "sigma": (0.04, 0.10),
    "alpha": (0.005, 0.02),
}
AMPLITUDE = 1.0


@dataclass
class CaseParams:
    x0: float
    y0: float
    sigma: float
    alpha: float

    def as_array(self):
        return np.array([self.x0, self.y0, self.sigma, self.alpha], dtype=np.float64)

    def validate(self):
        for name, (lo, hi) in PARAM_LIMITS.items():
            x = getattr(self, name)
            if not (lo <= x <= hi):
                raise ValueError(f"{name}={x} outside [{lo}, {hi}]")


def make_mesh_space(mesh_n=MESH_N):
    mesh = UnitSquareMesh(mesh_n, mesh_n)
    V = FunctionSpace(mesh, *ELEMENT)
    return mesh, V


def gaussian_ic(V, p: CaseParams):
    """Exact Gaussian initial condition as a Function in V."""
    x, y = SpatialCoordinate(V.mesh())
    r2 = (x - p.x0) ** 2 + (y - p.y0) ** 2
    return Function(V).interpolate(
        AMPLITUDE * exp(-r2 / (2 * p.sigma ** 2)))


def run_simulation(p: CaseParams, mesh_n=MESH_N):
    """Full transient solve; returns (times, solutions[Nt, Ndof], runtime_ms).

    solutions are in Firedrake's native dof ordering; use the canonical
    plot order from data/operators/order.npy for reshaping to 33x33.
    """
    p.validate()
    mesh, V = make_mesh_space(mesh_n)
    ndofs = V.dof_count

    u = Function(V, name="u")
    u_old = gaussian_ic(V, p)

    trial, test = TrialFunction(V), TestFunction(V)
    a = inner(trial, test) * dx + DT * p.alpha * inner(grad(trial), grad(test)) * dx
    L = inner(u_old, test) * dx
    problem = LinearVariationalProblem(a, L, u)
    solver = LinearVariationalSolver(problem)

    n_out = len(TIMES)
    steps_per_out = int(round(OUTPUT_DT / DT))
    solutions = np.empty((n_out, ndofs), dtype=np.float64)

    t0 = time.perf_counter()
    solutions[0] = u_old.dat.data_ro.copy()
    for k in range(1, n_out):
        for _ in range(steps_per_out):
            solver.solve()
            u_old.assign(u)
        solutions[k] = u.dat.data_ro.copy()
    runtime_ms = (time.perf_counter() - t0) * 1000.0

    return TIMES.copy(), solutions, runtime_ms


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Single Firedrake heat solve")
    parser.add_argument("--x0", type=float, default=0.40)
    parser.add_argument("--y0", type=float, default=0.60)
    parser.add_argument("--sigma", type=float, default=0.07)
    parser.add_argument("--alpha", type=float, default=0.01)
    parser.add_argument("--out", type=str, default=None,
                        help="optional .npz output path")
    args = parser.parse_args()

    p = CaseParams(args.x0, args.y0, args.sigma, args.alpha)
    times, sols, rt_ms = run_simulation(p)
    print(f"ndofs     : {sols.shape[1]}")
    print(f"frames    : {sols.shape[0]}")
    print(f"runtime   : {rt_ms:.1f} ms")

    # --- scientific sanity checks -------------------------------------
    peak = sols.max(axis=1)
    print(f"peak t=0  : {peak[0]:.6f}")
    print(f"peak t=end: {peak[-1]:.6f}")
    monotonically_decaying = bool(np.all(np.diff(peak) < 1e-12))
    print(f"peak decays monotonically: {monotonically_decaying}")

    # total heat  m(t) = 1^T M u(t)  requires M; approximate with trapezoidal
    # node-sum proxy is not exact, so only check smoothness here.
    d2 = np.abs(np.diff(sols, 2, axis=0)).max()
    print(f"max |second time difference|: {d2:.3e}")

    if args.out:
        np.savez_compressed(args.out, params=p.as_array(), times=times,
                            solutions=sols.astype(np.float32),
                            solver_runtime_ms=rt_ms)
        print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
