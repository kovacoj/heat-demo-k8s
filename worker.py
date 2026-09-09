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

# Direct solve for the mixed Cahn-Hilliard wax system:
# indefinite saddle-point matrix, factorized once.
SOLVER_PARAMS_DIRECT = {
    "mat_type": "aij",
    "ksp_type": "preonly",
    "pc_type": "lu",
    "pc_factor_mat_solver_type": "mumps",
}

# Lava lamp geometry: a tall, narrow glass.
DOMAIN_WIDTH = 1.0
DOMAIN_HEIGHT = 2.0

# Kick-start plumes and seed the wax: four balls with a
# ~30% volume fraction. Below ~20% the dissolved state wins
# energetically and Cahn-Hilliard (correctly) melts the
# balls; above it, droplets are the stable state.
# (x, y, radius of the wax balls)
BLOBS = (
    (0.35, 0.50, 0.22),
    (0.65, 0.75, 0.22),
    (0.40, 1.25, 0.22),
    (0.60, 1.50, 0.22),
)


def potential(x):
    # Quartic double-well: wells at 0 and 1.
    return (1 - x) ** 2 * x ** 2


def potential_derivative(x):
    return 2 * x * (1 - x) * (1 - 2 * x)


def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--ra",
        type=float,
        default=20000.0,
    )

    parser.add_argument(
        "--pr",
        type=float,
        default=10.0,
        help="Prandtl number: high = viscous wax-like fluid",
    )

    parser.add_argument(
        "--t-hot",
        type=float,
        default=2.0,
        help="bottom plate temperature (top and walls are 0)",
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

    parser.add_argument(
        "--ch-epsilon",
        type=float,
        default=0.06,
        help="Cahn-Hilliard interface width (wax ball sharpness)",
    )

    parser.add_argument(
        "--ch-mobility",
        type=float,
        default=4.0,
        help="Cahn-Hilliard mobility (wax coarsening speed)",
    )

    parser.add_argument(
        "--wax-buoyancy",
        type=float,
        default=0.05,
        help="extra thermal expansion of wax vs fluid (hot wax rises)",
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
    #   d(omega)/dt + u·grad(omega) = Pr ∇²(omega)
    #       + Pr Ra [dT/dx + W d(cT)/dx]
    #   dT/dt     + u·grad(T)     = ∇²T
    #   ∇²(psi) = -omega,  u = (d(psi)/dy, -d(psi)/dx)
    #
    # Hot bottom plate (T = t_hot), everything else cold
    # (T = 0): the fluid heats at the bottom, rises, cools
    # at the top and walls, and sinks back down.
    #
    # The rendered field is a "wax" phase field c governed
    # by Cahn-Hilliard, stirred by the flow:
    #
    #   dc/dt + u·grad(c) = M div(grad(mu))
    #   mu = f'(c) - eps² laplace(c),  f(c) = (1 - c)² c²
    #
    # The quartic double-well makes intermediate wax clump
    # together (spinodal decomposition) and the eps² term is
    # surface tension keeping the balls round and compact.
    # Cahn-Hilliard is in conservation form with no-flux
    # walls, so the total wax mass is conserved exactly.
    # f'(c) is taken from the previous step (linearized),
    # which keeps the mixed matrix constant — one LU
    # factorization for the whole run.
    #
    # Free-slip walls: psi = 0 and omega = 0 on the boundary.
    # Diffusion is treated implicitly (constant Jacobians),
    # advection explicitly (CFL-limited dt).
    #
    # W (wax buoyancy) makes hot wax rise and cold wax sink
    # like in a real lava lamp; without it all wax eventually
    # pools in a dead blob at the bottom.
    # -------------------------------------------------------

    Pr = fd.Constant(args.pr)
    Ra = fd.Constant(args.ra)
    dt = fd.Constant(args.dt)

    # State, updated in place each step.
    psi = fd.Function(V, name="psi")
    omega = fd.Function(V, name="omega")
    T = fd.Function(V, name="temperature")

    # Gaussian helper for seeding blobs.
    def gaussian(x0, y0, sigma):
        return fd.exp(
            -(
                (X - x0) ** 2
                + (Y - y0) ** 2
            )
            / (2.0 * sigma**2)
        )

    # Conduction profile plus warm anomalies that
    # kick-start rising plumes.
    T.interpolate(
        args.t_hot
        * (
            (1.0 - Y / DOMAIN_HEIGHT)
            +
            0.25
            * sum(
                gaussian(bx, by, bs)
                for bx, by, bs in BLOBS
            )
        )
        +
        0.01
        * args.t_hot
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
    #
    # Hot bottom plate, cold everywhere else.
    bcs_T = (
        fd.DirichletBC(V, args.t_hot, 3),
        fd.DirichletBC(V, 0.0, 4),
        fd.DirichletBC(V, 0.0, 1),
        fd.DirichletBC(V, 0.0, 2),
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

    # -------------------------------------------------------
    # Wax: Cahn-Hilliard phase field, linearized.
    #
    #   (c^{n+1} - c^n)/dt + u·grad(c^n)
    #       = -M (grad(mu^{n+1}), grad(.))
    #   mu^{n+1} = f'(c^n) - eps² laplace(c^{n+1})
    #
    # The potential derivative is taken from the previous
    # step, so the mixed (c, mu) matrix is constant and is
    # factorized once for the whole run. Advection and f'(c)
    # are explicit; their combined stability needs roughly
    # dt < eps² / (M · max|f''|).
    # No Dirichlet BCs: natural no-flux keeps total wax
    # conserved exactly.
    # -------------------------------------------------------

    Q_c = fd.FunctionSpace(mesh, "CG", 1)
    M_c = fd.FunctionSpace(mesh, "CG", 1)

    W_c = Q_c * M_c

    w_c = fd.Function(W_c)
    c, mu = w_c.subfunctions

    c_t, mu_t = fd.TrialFunctions(W_c)
    psi_c, nu_c = fd.TestFunctions(W_c)

    ch_eps = args.ch_epsilon
    ch_mobility = args.ch_mobility

    a_c = (
        c_t * psi_c
        +
        dt
        * ch_mobility
        * fd.dot(
            fd.grad(mu_t),
            fd.grad(psi_c),
        )
        +
        mu_t * nu_c
        -
        ch_eps**2
        * fd.dot(
            fd.grad(c_t),
            fd.grad(nu_c),
        )
    ) * fd.dx

    # c^n (advected explicitly by the current velocity),
    # updated in place before each wax solve.
    c_old = fd.Function(Q_c, name="wax_old")

    L_c = (
        c_old * psi_c
        -
        dt
        * fd.dot(
            u,
            fd.grad(c_old),
        )
        * psi_c
        +
        potential_derivative(c_old)
        * nu_c
    ) * fd.dx

    # -------------------------------------------------------
    # Initial conditions for the wax: four balls, with the
    # chemical potential consistent with the phase.
    # -------------------------------------------------------

    initial_wax = 0.0

    for bx, by, br in BLOBS:
        distance = fd.sqrt(
            (X - bx) ** 2
            + (Y - by) ** 2
            + 1.0e-12
        )
        initial_wax = initial_wax + 0.5 * (
            1.0
            - fd.tanh(
                (distance - br)
                / (0.5 * ch_eps)
            )
        )

    c.interpolate(initial_wax)

    mu_t0 = fd.TrialFunction(M_c)
    nu_0 = fd.TestFunction(M_c)

    a_mu = mu_t0 * nu_0 * fd.dx

    L_mu = (
        potential_derivative(c)
        * nu_0
        +
        ch_eps**2
        * fd.dot(
            fd.grad(c),
            fd.grad(nu_0),
        )
    ) * fd.dx

    fd.solve(
        a_mu == L_mu,
        mu,
        solver_parameters=SOLVER_PARAMS_DIRECT,
    )

    c_old.assign(c)
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
    #
    # Lava-lamp physics: wax expands more than the ambient
    # fluid when heated, so hot wax is extra buoyant and
    # rises, cools near the top and sinks again — the term
    # Ra_w · Dx(c·T, 0) below. c is the wax from the
    # previous wax step (one-step lag).
    w_t = fd.TrialFunction(V)

    wax_buoyancy = fd.Constant(args.wax_buoyancy)

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
        * (
            fd.Dx(T, 0)
            +
            wax_buoyancy
            * fd.Dx(c * T, 0)
        )
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

    solver_c = fd.LinearVariationalSolver(
        fd.LinearVariationalProblem(
            a_c,
            L_c,
            w_c,
            constant_jacobian=True,
        ),
        solver_parameters=SOLVER_PARAMS_DIRECT,
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
        #
        # We render the wax tracer c, not the temperature:
        # blobs of conserved material instead of diffusing heat.
        values = np.asarray(
            evaluator.evaluate(c)
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
        # then psi (next step's advection velocity),
        # then the wax (advected by the new velocity).
        solver_T.solve()
        solver_w.solve()
        solver_psi.solve()

        # Linearized CH: f'(c) and advection come from c^n.
        c_old.assign(c)
        solver_c.solve()

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
