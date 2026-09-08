import argparse
import json

import numpy as np
from mpi4py import MPI
import firedrake as fd


PROTOCOL_PREFIX = "NDJSON:"

SOLVER_PARAMS = {
    "mat_type": "aij",
    "ksp_type": "cg",
    "pc_type": "gamg",
    "ksp_rtol": 1.0e-8,
    "ksp_atol": 1.0e-12,
    "ksp_max_it": 200,
}

# Lava lamp geometry: a tall, narrow glass.
DOMAIN_WIDTH = 1.0
DOMAIN_HEIGHT = 2.0

# Kick-start plume: warm blob near the bottom.
BLOB_X = 0.5
BLOB_Y = 0.6
BLOB_SIGMA = 0.12


def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--ra",
        type=float,
        default=20000.0,
    )

    parser.add_argument(
        "--mesh-n",
        type=int,
        default=64,
        help="cells across the width (height gets twice as many)",
    )

    parser.add_argument(
        "--sample-n",
        type=int,
        default=64,
        help="sampling points across the width (height gets twice as many)",
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
    ys = np.linspace(0.0, DOMAIN_HEIGHT, 2 * n)

    return np.column_stack(
        (
            np.tile(xs, 2 * n),
            np.repeat(ys, n),
        )
    )


def run_lavalamp(args, comm):

    emit(
        comm,
        {
            "type": "status",
            "message": (
                f"Creating {args.mesh_n}×{2 * args.mesh_n} "
                f"parallel Firedrake mesh..."
            ),
        },
    )

    # -------------------------------------------------------
    # Mesh: tall glass of hot liquid, heated from below.
    # -------------------------------------------------------

    mesh = fd.RectangleMesh(
        args.mesh_n,
        2 * args.mesh_n,
        DOMAIN_WIDTH,
        DOMAIN_HEIGHT,
    )

    V = fd.FunctionSpace(
        mesh,
        "CG",
        1,
    )

    X, Y = fd.SpatialCoordinate(mesh)

    emit(
        comm,
        {
            "type": "status",
            "message": (
                "Building Boussinesq convection solver "
                "(streamfunction–vorticity)..."
            ),
        },
    )

    # -------------------------------------------------------
    # Nondimensional 2D Rayleigh–Bénard convection,
    # streamfunction–vorticity form:
    #
    #   d(omega)/dt + u·grad(omega) = Pr ∇²(omega) + Pr Ra dT/dx
    #   dT/dt     + u·grad(T)     = ∇²T
    #   ∇²(psi) = -omega,  u = (d(psi)/dy, -d(psi)/dx)
    #
    # Hot bottom plate (T = 1), cold top plate (T = 0),
    # insulating side walls. Fluid heats up at the bottom,
    # rises, cools at the top, sinks back down, repeats.
    #
    # Free-slip walls: psi = 0 and omega = 0 on the boundary.
    # Diffusion is treated implicitly (constant Jacobians),
    # advection explicitly (CFL-limited dt).
    # -------------------------------------------------------

    Pr = 1.0
    Ra = fd.Constant(args.ra)
    dt = fd.Constant(args.dt)

    # State, updated in place each step.
    psi = fd.Function(V, name="psi")
    omega = fd.Function(V, name="omega")
    T = fd.Function(V, name="temperature")

    # Conduction profile plus a warm blob that
    # kick-starts the first rising plume.
    T.interpolate(
        (1.0 - Y / DOMAIN_HEIGHT)
        +
        0.25
        * fd.exp(
            -(
                (X - BLOB_X) ** 2
                + (Y - BLOB_Y) ** 2
            )
            / (2.0 * BLOB_SIGMA**2)
        )
        +
        0.01
        * fd.sin(37.2 * X + 1.3)
        * fd.sin(18.5 * Y)
    )

    bc_psi = fd.DirichletBC(
        V,
        0.0,
        "on_boundary",
    )

    bc_omega = fd.DirichletBC(
        V,
        0.0,
        "on_boundary",
    )

    # RectangleMesh boundary IDs (probed empirically —
    # they differ from UnitSquareMesh!):
    # 1 = left (x=0), 2 = right (x=1),
    # 3 = bottom (y=0), 4 = top (y=H).
    bcs_T = (
        fd.DirichletBC(V, 1.0, 3),
        fd.DirichletBC(V, 0.0, 4),
    )

    v = fd.TestFunction(V)

    # Velocity from the streamfunction.
    u = fd.as_vector(
        [
            fd.Dx(psi, 1),
            -fd.Dx(psi, 0),
        ]
    )

    # Diagnostics: max |u| for display.
    W = fd.VectorFunctionSpace(mesh, "CG", 1)
    u_diag = fd.Function(W, name="velocity")

    def max_speed():

        u_diag.interpolate(u)

        data = u_diag.dat.data_ro

        local = (
            float(
                np.sqrt(
                    (data ** 2).sum(axis=-1)
                ).max()
            )
            if data.size
            else 0.0
        )

        return comm.allreduce(local, op=MPI.MAX)

    # Temperature: implicit diffusion, explicit advection.
    T_t = fd.TrialFunction(V)

    a_T = (
        T_t * v
        +
        dt
        * fd.dot(
            fd.grad(T_t),
            fd.grad(v),
        )
    ) * fd.dx

    L_T = (
        T
        -
        dt
        * fd.dot(
            u,
            fd.grad(T),
        )
    ) * v * fd.dx

    # Vorticity: implicit diffusion, explicit advection
    # plus buoyancy (uses the freshly updated T).
    w_t = fd.TrialFunction(V)

    a_w = (
        w_t * v
        +
        dt
        * Pr
        * fd.dot(
            fd.grad(w_t),
            fd.grad(v),
        )
    ) * fd.dx

    L_w = (
        omega
        -
        dt
        * fd.dot(
            u,
            fd.grad(omega),
        )
        +
        dt
        * Pr
        * Ra
        * fd.Dx(T, 0)
    ) * v * fd.dx

    # Streamfunction: ∇²(psi) = -omega.
    #
    # Weak form: ∫∇psi·∇v = ∫(+omega)·v  — note the PLUS
    # sign (integrating -∇²psi = -omega by parts flips it).
    p_t = fd.TrialFunction(V)

    a_psi = (
        fd.dot(
            fd.grad(p_t),
            fd.grad(v),
        )
        * fd.dx
    )

    L_psi = (
        omega
    ) * v * fd.dx

    solver_T = fd.LinearVariationalSolver(
        fd.LinearVariationalProblem(
            a_T,
            L_T,
            T,
            bcs=bcs_T,
            constant_jacobian=True,
        ),
        solver_parameters=SOLVER_PARAMS,
    )

    solver_w = fd.LinearVariationalSolver(
        fd.LinearVariationalProblem(
            a_w,
            L_w,
            omega,
            bcs=bc_omega,
            constant_jacobian=True,
        ),
        solver_parameters=SOLVER_PARAMS,
    )

    solver_psi = fd.LinearVariationalSolver(
        fd.LinearVariationalProblem(
            a_psi,
            L_psi,
            psi,
            bcs=bc_psi,
            constant_jacobian=True,
        ),
        solver_parameters=SOLVER_PARAMS,
    )

    # -------------------------------------------------------
    # Browser sampling grid
    # -------------------------------------------------------

    emit(
        comm,
        {
            "type": "status",
            "message": (
                f"Building {args.sample_n}×{2 * args.sample_n} "
                f"visualization grid..."
            ),
        },
    )

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

            "grid_y": 2 * args.sample_n,

            "dt": args.dt,

            "ra": args.ra,

            "mpi_ranks": comm.size,
        },
    )

    # -------------------------------------------------------
    # Frame helper
    # -------------------------------------------------------

    def send_frame(step, solve_seconds=None):

        # All MPI ranks call evaluate(); PointEvaluator
        # restores the requested point ordering.
        values = np.asarray(
            evaluator.evaluate(T)
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

        # Collective (allreduce): must run on ALL ranks,
        # never only inside the rank-0 branch below.
        speed = max_speed()

        if comm.rank == 0:

            message = {
                "type": "frame",

                "step": step,

                "time": step * args.dt,

                "min": float(values.min()),

                "max": float(values.max()),

                "max_speed": speed,

                "values": [
                    round(val, 4) for val in values.tolist()
                ],
            }

            if solve_seconds is not None:
                message["solve_seconds"] = solve_seconds

            emit(
                comm,
                message,
            )

        return True

    # Initial condition
    send_frame(step=0)

    emit(
        comm,
        {
            "type": "status",
            "message": "Lamp running.",
        },
    )

    # -------------------------------------------------------
    # Time stepping
    # -------------------------------------------------------

    for step in range(
        1,
        args.steps + 1,
    ):

        start = MPI.Wtime()

        # Order matters: T first (buoyancy uses the new T),
        # then omega (psi RHS uses the new omega),
        # then psi (next step's advection velocity).
        solver_T.solve()
        solver_w.solve()
        solver_psi.solve()

        local_elapsed = (
            MPI.Wtime() - start
        )

        solve_seconds = comm.reduce(
            local_elapsed,
            op=MPI.MAX,
            root=0,
        )

        should_stream = (
            step % args.stream_every == 0
        )

        if should_stream:

            if not send_frame(
                step=step,
                solve_seconds=solve_seconds,
            ):

                break

    emit(
        comm,
        {
            "type": "done",
            "steps": args.steps,
            "time": (
                args.steps
                * args.dt
            ),
        },
    )


def main():

    args = parse_args()

    comm = MPI.COMM_WORLD

    run_lavalamp(args, comm)


if __name__ == "__main__":
    main()
