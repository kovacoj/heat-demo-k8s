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
    "snes_rtol": 1.0e-8,
    "snes_atol": 1.0e-8,
    "ksp_type": "gmres",
    "ksp_max_it": 1000,
    "pc_type": "fieldsplit",
    "pc_fieldsplit_type": "schur",
    "pc_fieldsplit_schur_fact_type": "full",
    "pc_fieldsplit_0_fields": "0,1",
    "pc_fieldsplit_1_fields": "2,3",
    "fieldsplit_0": {
        "ksp_type": "preonly",
        "pc_type": "lu",
        "pc_factor_mat_solver_type": "mumps",
        "pc_factor_shift_type": "nonzero",
        "mat_mumps_icntl_14": 500,
    },
    "fieldsplit_1": {
        "ksp_type": "gmres",
        "pc_type": "hypre",
        "pc_hypre_type": "boomeramg",
        "ksp_rtol": 1.0e-2,
    },
}

# Domain: tall box, 1 x 3.
DOMAIN_WIDTH = 1.0
DOMAIN_HEIGHT = 3.0

# CHNS parameters (from the ../chns project).
RHO1, RHO2 = 10.0, 1.0
NU1 = NU2 = 1.0
SIGMA = 1.0e-1
EPSILON = 1.0e-1
M0 = 1.0e-4

# Fully implicit in everything but kept structured like
# the original theta-scheme.
THETA = 1.0

# Two bubbles rising and merging.
BUBBLES = (
    (0.5, 0.65, 0.14),
    (0.5, 1.05, 0.14),
)


def potential(x):
    return (1 - x) ** 2 * x ** 2


def potential_derivative(x):
    return 2 * x * (1 - x) * (1 - 2 * x)


def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--mesh-n",
        type=int,
        default=20,
        help="cells across the width (height gets 3x as many)",
    )

    parser.add_argument(
        "--sample-n",
        type=int,
        default=24,
        help="sampling points across the width",
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
    xs = np.linspace(0.0, DOMAIN_WIDTH, n)
    ys = np.linspace(0.0, DOMAIN_HEIGHT, 3 * n)

    return np.column_stack(
        (
            np.tile(xs, 3 * n),
            np.repeat(ys, n),
        )
    )


def run_chns(args, comm):

    emit(
        comm,
        {
            "type": "status",
            "message": (
                f"Creating {args.mesh_n}×{3 * args.mesh_n} "
                f"parallel Firedrake mesh..."
            ),
        },
    )

    mesh = fd.RectangleMesh(
        args.mesh_n,
        3 * args.mesh_n,
        DOMAIN_WIDTH,
        DOMAIN_HEIGHT,
    )

    # Taylor-Hood velocity-pressure pair with scalar CG1
    # phase and chemical potential spaces.
    k = 2

    V = fd.VectorFunctionSpace(mesh, "CG", k)
    P = fd.FunctionSpace(mesh, "CG", k - 1)
    Q = fd.FunctionSpace(mesh, "CG", k - 1)
    M = fd.FunctionSpace(mesh, "CG", k - 1)

    W = V * P * Q * M

    emit(
        comm,
        {
            "type": "status",
            "message": "Building Cahn-Hilliard-Navier-Stokes solver...",
        },
    )

    # -------------------------------------------------------
    # Cahn-Hilliard-Navier-Stokes (from ../chns):
    #
    #   rho(phi) [du/dt + (u·grad)u]
    #       = div(nu(phi) grad(u)) - grad(p)
    #         + (rho(phi) - rho1) g + mu grad(phi)
    #   div(u) = 0
    #   d(phi)/dt + u·grad(phi) = div(m(phi) grad(mu))
    #   mu = -eps*sigma*laplace(phi) + sigma/eps * f'(phi)
    #
    # The light fluid (phi = 1, rho2) rises as buoyant
    # bubbles through the heavy fluid (phi = 0, rho1).
    # No-slip walls.
    # -------------------------------------------------------

    dt = fd.Constant(args.dt)

    gravity = fd.Constant((0.0, -9.81))

    w = fd.Function(W)
    w_ = fd.Function(W)

    u, p, phi, mu = fd.split(w)
    u_, p_, phi_, mu_ = fd.split(w_)

    v, q, psi, nu = fd.TestFunctions(W)

    def density(phase):
        return fd.conditional(
            phase < 0,
            RHO1,
            fd.conditional(
                phase > 1,
                RHO2,
                RHO1 + phase * (RHO2 - RHO1),
            ),
        )

    def viscosity(phase):
        return NU2 * phase + NU1 * (1.0 - phase)

    def mobility(phase):
        return M0 * potential(phase) + 1.0e-6

    def buoyancy(phase):
        return (density(phase) - RHO1) * gravity

    def momentum(u, p, phi, mu):
        return (
            density(phi)
            * fd.inner(
                fd.dot(u, fd.nabla_grad(u)),
                v,
            )
            + viscosity(phi)
            * fd.inner(fd.grad(u), fd.grad(v))
            - fd.dot(buoyancy(phi), v)
            - p * fd.div(v)
            + mu * fd.inner(fd.grad(phi), v)
        )

    def phase(u, p, phi, mu):
        return (
            fd.inner(
                fd.dot(u, fd.grad(phi)),
                psi,
            )
            + mobility(phi)
            * fd.inner(fd.grad(mu), fd.grad(psi))
        )

    F = (
        fd.inner((phi - phi_) / dt, psi)
        + fd.inner(
            (
                density(phi) * u
                - density(phi_) * u_
            ) / dt,
            v,
        )
        + THETA * momentum(u, p, phi, mu)
        + THETA * phase(u, p, phi, mu)
        + (1.0 - THETA) * momentum(u_, p_, phi_, mu_)
        + (1.0 - THETA) * phase(u_, p_, phi_, mu_)
        + q * fd.div(u)
        + fd.inner(mu, nu)
        - EPSILON * SIGMA
        * fd.dot(fd.grad(phi), fd.grad(nu))
        - SIGMA / EPSILON
        * potential_derivative(phi)
        * nu
    ) * fd.dx

    bcs = [
        fd.DirichletBC(
            W.sub(0),
            fd.Constant((0.0, 0.0)),
            "on_boundary",
        ),
    ]

    pressure_nullspace = fd.MixedVectorSpaceBasis(
        W,
        [
            W.sub(0),
            fd.VectorSpaceBasis(
                constant=True,
                comm=mesh.comm,
            ),
            W.sub(2),
            W.sub(3),
        ],
    )

    # -------------------------------------------------------
    # Initial conditions: two bubbles at rest.
    # -------------------------------------------------------

    x, y = fd.SpatialCoordinate(mesh)

    interface_width = 0.5 * EPSILON

    initial_phase = 0.0

    for bx, by, br in BUBBLES:
        distance = fd.sqrt(
            (x - bx) ** 2 + (y - by) ** 2 + 1.0e-12
        )
        initial_phase = initial_phase + 0.5 * (
            1.0
            - fd.tanh(
                (distance - br) / interface_width
            )
        )

    u_fn, p_fn, phi_fn, mu_fn = w.subfunctions

    phi_fn.interpolate(initial_phase)

    # Consistent initial chemical potential: the weak form
    # is linear in mu for a given phi.
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

    problem = fd.NonlinearVariationalProblem(
        F,
        w,
        bcs=bcs,
    )

    solver = fd.NonlinearVariationalSolver(
        problem,
        solver_parameters=SOLVER_PARAMS,
        nullspace=pressure_nullspace,
        transpose_nullspace=pressure_nullspace,
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

            "grid_y": 3 * args.sample_n,

            "dt": args.dt,

            "system": "chns",

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

    run_chns(args, comm)


if __name__ == "__main__":
    main()
