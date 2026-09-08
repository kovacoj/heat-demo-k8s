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


def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--mode",
        choices=["diffusion", "convection", "lavalamp"],
        default="diffusion",
    )

    parser.add_argument(
        "--ra",
        type=float,
        default=2000.0,
    )

    parser.add_argument(
        "--x",
        type=float,
        required=True,
    )

    parser.add_argument(
        "--y",
        type=float,
        required=True,
    )

    parser.add_argument(
        "--sigma",
        type=float,
        required=True,
    )

    parser.add_argument(
        "--diffusivity",
        type=float,
        required=True,
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
        "--mesh-n",
        type=int,
        required=True,
    )

    parser.add_argument(
        "--sample-n",
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


def sample_points(n):

    xs = np.linspace(0.0, 1.0, n)
    ys = np.linspace(0.0, 1.0, n)

    return np.column_stack(
        (
            np.tile(xs, n),
            np.repeat(ys, n),
        )
    )


def run_diffusion(args, comm):

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
    # Mesh
    # -------------------------------------------------------

    mesh = fd.UnitSquareMesh(
        args.mesh_n,
        args.mesh_n,
    )

    V = fd.FunctionSpace(
        mesh,
        "CG",
        1,
    )

    X, Y = fd.SpatialCoordinate(mesh)

    # -------------------------------------------------------
    # Initial condition
    # -------------------------------------------------------

    T_old = fd.Function(
        V,
        name="temperature_old",
    )

    T_old.interpolate(
        fd.exp(
            -(
                (X - args.x) ** 2
                + (Y - args.y) ** 2
            )
            / (2.0 * args.sigma**2)
        )
    )

    bc = fd.DirichletBC(
        V,
        0.0,
        "on_boundary",
    )

    bc.apply(T_old)

    # -------------------------------------------------------
    # Crank-Nicolson
    # -------------------------------------------------------

    emit(
        comm,
        {
            "type": "status",
            "message": "Building Crank–Nicolson solver...",
        },
    )

    u = fd.TrialFunction(V)
    v = fd.TestFunction(V)

    T_new = fd.Function(
        V,
        name="temperature",
    )

    dt = fd.Constant(args.dt)

    alpha = fd.Constant(
        args.diffusivity
    )

    # M + dt/2 * alpha * K
    a = (
        u * v * fd.dx
        +
        0.5
        * dt
        * alpha
        * fd.inner(
            fd.grad(u),
            fd.grad(v),
        )
        * fd.dx
    )

    # (M - dt/2 * alpha * K) T_old
    L = (
        T_old * v * fd.dx
        -
        0.5
        * dt
        * alpha
        * fd.inner(
            fd.grad(T_old),
            fd.grad(v),
        )
        * fd.dx
    )

    problem = fd.LinearVariationalProblem(
        a,
        L,
        T_new,
        bcs=bc,

        # Matrix does not change between timesteps.
        constant_jacobian=True,
    )

    solver = fd.LinearVariationalSolver(
        problem,
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
                f"Building {args.sample_n}×{args.sample_n} "
                f"visualization grid..."
            ),
        },
    )

    points = sample_points(args.sample_n)

    evaluator = fd.PointEvaluator(
        mesh,
        points,
    )

    emit(
        comm,
        {
            "type": "meta",

            "mode": "diffusion",

            "mesh_n": args.mesh_n,

            "sample_n": args.sample_n,

            "steps": args.steps,

            "stream_every": args.stream_every,

            "dt": args.dt,

            "diffusivity": args.diffusivity,

            "mpi_ranks": comm.size,
        },
    )

    # -------------------------------------------------------
    # Frame helper
    # -------------------------------------------------------

    def send_frame(field, step, solve_seconds=None):

        # IMPORTANT:
        #
        # All MPI ranks call evaluate().
        # PointEvaluator handles parallel ownership and
        # returns the points in input order.
        values = np.asarray(
            evaluator.evaluate(field)
        ).reshape(-1)

        if comm.rank == 0:

            message = {
                "type": "frame",

                "step": step,

                "time": step * args.dt,

                "min": float(values.min()),

                "max": float(values.max()),

                "values": values.tolist(),
            }

            if solve_seconds is not None:
                message["solve_seconds"] = solve_seconds

            emit(
                comm,
                message,
            )

    # Initial condition
    send_frame(
        T_old,
        step=0,
    )

    emit(
        comm,
        {
            "type": "status",
            "message": "Simulation running.",
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

        solver.solve()

        local_elapsed = (
            MPI.Wtime() - start
        )

        # Slowest rank determines wall-clock solve time.
        solve_seconds = comm.reduce(
            local_elapsed,
            op=MPI.MAX,
            root=0,
        )

        should_stream = (
            step % args.stream_every == 0
            or step == args.steps
        )

        if should_stream:

            send_frame(
                T_new,
                step=step,
                solve_seconds=solve_seconds,
            )

        T_old.assign(T_new)

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


def run_convection(args, comm):

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
    # Mesh
    # -------------------------------------------------------

    mesh = fd.UnitSquareMesh(
        args.mesh_n,
        args.mesh_n,
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
    # Free-slip walls: psi = 0 and omega = 0 on the boundary.
    # Hot bottom (T = 1), cold top (T = 0), insulating sides.
    #
    # Diffusion is treated implicitly (constant Jacobians),
    # advection explicitly (CFL-limited dt, see app.py).
    # -------------------------------------------------------

    Ra = fd.Constant(args.ra)
    dt = fd.Constant(args.dt)

    # Thermal-time nondimensionalization (same as the
    # Rayleigh–Bénard mode): Pr = 1. A high enough Ra
    # keeps rising blobs coherent against diffusion.
    kappa = 1.0  # thermal diffusivity
    nu = 1.0     # momentum diffusivity
    buoy = Ra    # buoyancy coefficient

    # State, updated in place each step.
    psi = fd.Function(V, name="psi")
    omega = fd.Function(V, name="omega")
    T = fd.Function(V, name="temperature")

    if args.mode == "lavalamp":

        # A lava lamp: cold glass walls, one heating
        # element near the bottom (like the bulb).
        # Warm fluid rises over the element, cools
        # against the glass and sinks back down.
        #
        # Start pre-warmed over the element so the
        # plume develops immediately.
        T.interpolate(
            0.7
            * fd.exp(
                -(
                    (X - args.x) ** 2
                    + (Y - 0.2) ** 2
                )
                / (2.0 * args.sigma**2)
            )
            +
            0.01
            * fd.sin(37.2 * X + 1.3)
            * fd.sin(41.7 * Y)
        )

    else:

        # Rayleigh–Bénard: conduction profile plus a
        # warm blob that kick-starts a rising plume.
        T.interpolate(
            (1.0 - Y)
            +
            0.25
            * fd.exp(
                -(
                    (X - args.x) ** 2
                    + (Y - 0.25) ** 2
                )
                / (2.0 * args.sigma**2)
            )
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

    if args.mode == "lavalamp":

        # Cold glass everywhere.
        bcs_T = fd.DirichletBC(
            V,
            0.0,
            "on_boundary",
        )

    else:

        # UnitSquareMesh boundary IDs:
        # 1 = bottom, 2 = right, 3 = top, 4 = left.
        bcs_T = (
            fd.DirichletBC(V, 1.0, 1),
            fd.DirichletBC(V, 0.0, 3),
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
        * kappa
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

    if args.mode == "lavalamp":

        # The heating element (the "bulb" of the lamp).
        S = 100.0 * fd.exp(
            -(
                (X - args.x) ** 2
                + (Y - 0.18) ** 2
            )
            / (2.0 * args.sigma**2)
        )

        L_T = (
            T
            -
            dt
            * fd.dot(
                u,
                fd.grad(T),
            )
            +
            dt * S
        ) * v * fd.dx

    # Vorticity: implicit diffusion, explicit advection
    # plus buoyancy (uses the freshly updated T).
    w_t = fd.TrialFunction(V)

    a_w = (
        w_t * v
        +
        dt
        * nu
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
        * buoy
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
                f"Building {args.sample_n}×{args.sample_n} "
                f"visualization grid..."
            ),
        },
    )

    points = sample_points(args.sample_n)

    evaluator = fd.PointEvaluator(
        mesh,
        points,
    )

    emit(
        comm,
        {
            "type": "meta",

            "mode": args.mode,

            "mesh_n": args.mesh_n,

            "sample_n": args.sample_n,

            "steps": args.steps,

            "stream_every": args.stream_every,

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
                        "Simulation diverged (NaN/inf). "
                        "Try a lower Rayleigh number."
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

                "values": values.tolist(),
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
            "message": "Simulation running.",
        },
    )

    # -------------------------------------------------------
    # Time stepping
    # -------------------------------------------------------

    diverged = False

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
            or step == args.steps
        )

        if should_stream:

            if not send_frame(
                step=step,
                solve_seconds=solve_seconds,
            ):

                diverged = True
                break

    if not diverged:

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

    if args.mode in ("convection", "lavalamp"):
        run_convection(args, comm)
    else:
        run_diffusion(args, comm)


if __name__ == "__main__":
    main()
