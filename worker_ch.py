import argparse
import json

import numpy as np
from mpi4py import MPI
import firedrake as fd


PROTOCOL_PREFIX = "NDJSON:"

SOLVER_PARAMS = {
    "snes_type": "newtonls",
    "snes_linesearch_type": "bt",
    "snes_max_it": 50,
    "snes_rtol": 1.0e-6,
    "snes_atol": 1.0e-6,
    "ksp_type": "preonly",
    "pc_type": "lu",
    "pc_factor_mat_solver_type": "mumps",
}

# Cahn-Hilliard parameters (from the ../chns project).
DOMAIN_SIZE = 1.0

SIGMA = 1.0e-1
EPSILON = 1.0e-1

# Crank-Nicolson parameter.
THETA = 0.5


def potential(x):
    return (1 - x) ** 2 * x ** 2


def potential_derivative(x):
    return 2 * x * (1 - x) * (1 - 2 * x)


def mobility(phase):
    return potential(phase) + 1.0e-3


def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--mesh-n",
        type=int,
        default=48,
        help="cells per side of the unit square",
    )

    parser.add_argument(
        "--sample-n",
        type=int,
        default=64,
        help="sampling points per side",
    )

    parser.add_argument(
        "--dt",
        type=float,
        required=True,
    )

    parser.add_argument(
        "--steps",
        type=int,
        required=True,
    )

    parser.add_argument(
        "--stream-every",
        type=int,
        required=True,
    )

    return parser.parse_args()


def emit(comm, message):

    if comm.rank == 0:

        print(
            PROTOCOL_PREFIX + json.dumps(
                message,
                separators=(",", ":"),
            ),
            flush=True,
        )


def sample_grid(n):

    # Ordering: y = 0 first, row-major within each y row.
    xs = np.linspace(0.0, DOMAIN_SIZE, n)
    ys = np.linspace(0.0, DOMAIN_SIZE, n)

    return np.column_stack(
        (
            np.tile(xs, n),
            np.repeat(ys, n),
        )
    )


def run_cahn_hilliard(args, comm):

    emit(
        comm,
        {
            "type": "status",
            "message": (
                f"Creating {args.mesh_n}×{args.mesh_n} "
                f"parallel Firedrake mesh..."
            ),
        },
    )

    # -------------------------------------------------------
    # Mesh: unit square. Cahn-Hilliard needs no boundary
    # conditions: the natural no-flux ones apply, and the
    # total phase mass is conserved exactly.
    # -------------------------------------------------------

    mesh = fd.RectangleMesh(
        args.mesh_n,
        args.mesh_n,
        DOMAIN_SIZE,
        DOMAIN_SIZE,
    )

    Q = fd.FunctionSpace(mesh, "CG", 1)
    M = fd.FunctionSpace(mesh, "CG", 1)

    W = Q * M

    emit(
        comm,
        {
            "type": "status",
            "message": "Building Cahn-Hilliard solver...",
        },
    )

    # -------------------------------------------------------
    # Cahn-Hilliard, mixed formulation (from ../chns):
    #
    #   d(phi)/dt = div(m(phi) grad(mu))
    #   mu = -eps*sigma*laplace(phi) + sigma/eps * f'(phi)
    #
    # with the quartic double-well potential
    #
    #   f(phi) = (1 - phi)^2 phi^2
    #
    # and degenerate mobility m(phi) = f(phi) + 1e-3.
    # Time discretization: Crank-Nicolson on the phase
    # equation, fully implicit chemical potential.
    # -------------------------------------------------------

    dt = fd.Constant(args.dt)

    w = fd.Function(W)
    w_ = fd.Function(W)

    phi, mu = fd.split(w)
    phi_, mu_ = fd.split(w_)

    psi, nu = fd.TestFunctions(W)

    # Initial phase: a handful of seeded bubbles that
    # coarsen and merge over time (fixed seed).
    x, y = fd.SpatialCoordinate(mesh)

    rng = np.random.default_rng(42)

    n_bubbles = 8
    radius = 0.1
    centers = []

    while len(centers) < n_bubbles:
        candidate = (
            rng.uniform(0.15, 0.85),
            rng.uniform(0.15, 0.85),
        )

        if all(
            (candidate[0] - cx) ** 2 + (candidate[1] - cy) ** 2
            > (2.6 * radius) ** 2
            for cx, cy in centers
        ):
            centers.append(candidate)

    interface_width = 0.5 * EPSILON

    initial_phase = 0.0

    for center_x, center_y in centers:
        r = radius * rng.uniform(0.8, 1.2)
        distance = fd.sqrt(
            (x - center_x) ** 2
            + (y - center_y) ** 2
            + 1.0e-12
        )
        initial_phase = initial_phase + 0.5 * (
            1.0
            - fd.tanh(
                (distance - r) / interface_width
            )
        )

    phi_fn, mu_fn = w.subfunctions

    phi_fn.interpolate(initial_phase)

    # The chemical potential must be initialized consistently
    # with the initial phase (from ../chns): the weak form is
    # linear in mu, so one direct solve suffices.
    mu_t = fd.TrialFunction(M)
    nu_0 = fd.TestFunction(M)

    a_mu = fd.inner(mu_t, nu_0) * fd.dx

    L_mu = (
        EPSILON * SIGMA
        * fd.dot(fd.grad(phi_fn), fd.grad(nu_0))
        +
        SIGMA / EPSILON
        * potential_derivative(phi_fn)
        * nu_0
    ) * fd.dx

    fd.solve(
        a_mu == L_mu,
        mu_fn,
        solver_parameters={
            "ksp_type": "preonly",
            "pc_type": "lu",
            "pc_factor_mat_solver_type": "mumps",
        },
    )

    w_.assign(w)

    # Crank-Nicolson on the phase equation.
    phase_residual = (
        lambda p, m: mobility(p)
        * fd.dot(fd.grad(m), fd.grad(psi))
    )

    F = (
        fd.inner((phi - phi_) / dt, psi)
        + THETA * phase_residual(phi, mu)
        + (1.0 - THETA) * phase_residual(phi_, mu_)
        + fd.inner(mu, nu)
        - EPSILON * SIGMA
        * fd.dot(fd.grad(phi), fd.grad(nu))
        - SIGMA / EPSILON
        * potential_derivative(phi)
        * nu
    ) * fd.dx

    problem = fd.NonlinearVariationalProblem(F, w)
    solver = fd.NonlinearVariationalSolver(
        problem,
        solver_parameters=SOLVER_PARAMS,
    )

    # -------------------------------------------------------
    # Browser sampling grid
    # -------------------------------------------------------

    points = sample_grid(args.sample_n)

    evaluator = fd.PointEvaluator(
        mesh,
        points,
    )

    emit(
        comm,
        {
            "type": "meta",

            "grid_x": args.sample_n,

            "grid_y": args.sample_n,

            "dt": args.dt,

            "system": "cahn-hilliard",

            "mpi_ranks": comm.size,
        },
    )

    # -------------------------------------------------------
    # Frame helper
    # -------------------------------------------------------

    def send_frame(step):

        # All MPI ranks call evaluate(); PointEvaluator
        # restores the requested point ordering.
        values = np.asarray(
            evaluator.evaluate(phi_fn)
        ).reshape(-1)

        finite = (
            bool(np.all(np.isfinite(values)))
            if comm.rank == 0
            else None
        )

        finite = comm.bcast(finite, root=0)

        if not finite:

            emit(
                comm,
                {
                    "type": "error",
                    "message": (
                        "Simulation diverged (NaN/inf)."
                    ),
                },
            )

            return False

        if comm.rank == 0:

            emit(
                comm,
                {
                    "type": "frame",

                    "step": step,

                    "time": step * args.dt,

                    "min": float(values.min()),

                    "max": float(values.max()),

                    "values": [
                        round(val, 4)
                        for val in values.tolist()
                    ],
                },
            )

        return True

    send_frame(step=0)

    # -------------------------------------------------------
    # Time stepping
    # -------------------------------------------------------

    for step in range(1, args.steps + 1):

        solver.solve()

        w_.assign(w)

        if step % args.stream_every == 0:

            if not send_frame(step=step):
                break

    emit(
        comm,
        {
            "type": "done",
            "steps": args.steps,
            "time": args.steps * args.dt,
        },
    )


def main():

    args = parse_args()

    comm = MPI.COMM_WORLD

    run_cahn_hilliard(args, comm)


if __name__ == "__main__":
    main()
