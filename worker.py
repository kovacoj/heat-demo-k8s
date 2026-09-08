import argparse
import json

import numpy as np
from mpi4py import MPI
import firedrake as fd


PROTOCOL_PREFIX = "NDJSON:"


def parse_args():

    parser = argparse.ArgumentParser()

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


def main():

    args = parse_args()

    comm = MPI.COMM_WORLD

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
        solver_parameters={
            "mat_type": "aij",

            "ksp_type": "cg",

            "pc_type": "gamg",

            "ksp_rtol": 1.0e-8,

            "ksp_atol": 1.0e-12,

            "ksp_max_it": 200,
        },
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

    xs = np.linspace(
        0.0,
        1.0,
        args.sample_n,
    )

    ys = np.linspace(
        0.0,
        1.0,
        args.sample_n,
    )

    # Ordering:
    #
    # y=0: x0, x1, ..., xn
    # y=1: ...
    #
    points = np.column_stack(
        (
            np.tile(
                xs,
                args.sample_n,
            ),

            np.repeat(
                ys,
                args.sample_n,
            ),
        )
    )

    evaluator = fd.PointEvaluator(
        mesh,
        points,
    )

    emit(
        comm,
        {
            "type": "meta",

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


if __name__ == "__main__":
    main()
