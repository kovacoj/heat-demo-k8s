"""OpenFOAM heat-equation demo for the Kubernetes lava lamp.

Runs OpenFOAM's minimal solver (laplacianFoam, the plain heat
equation) on a 2D plate in parallel:

    dT/dt = div( alpha * grad(T) )

This script is the job's main process. It is NOT an MPI rank:
it prepares the case, launches `mpirun -np R laplacianFoam
-parallel` as a subprocess, and supervises it:

- writes the case from templates (blockMesh, setFields,
  decomposePar),
- tails each rank's `probes` output (a 64x64 point grid sampled
  from the temperature field),
- merges the per-rank probe values into full frames and emits
  them as NDJSON (same protocol as the Firedrake workers),
- paces the solver with SIGSTOP/SIGCONT so the wall-clock
  speed is constant regardless of cluster load,
- when a run reaches its end, regenerates the case with fresh
  random hot blobs and starts over — an endless demo.

Boundary conditions: hot left wall (500 K), cold right wall
(300 K), insulated top and bottom. Values sent to the browser
are normalized to [0, 1].
"""

import argparse
import json
import os
import random
import shutil
import signal
import subprocess
import sys
import time


PROTOCOL_PREFIX = "NDJSON:"

CASE_DIR = "/tmp/of-case"

# Physical temperature range [K] -> normalized [0, 1] for the
# browser palette.
T_MIN = 300.0
T_MAX = 500.0
T_BASE = 350.0

# Plate thickness (one cell): makes the case 2D.
THICKNESS = 0.01

# Directory where OpenFOAM's bashrc lives, e.g.
# /usr/lib/openfoam/openfoam2512/etc/bashrc
OF_GLOB = "/usr/lib/openfoam/openfoam*/etc/bashrc"


def emit(message):
    print(
        PROTOCOL_PREFIX + json.dumps(
            message,
            separators=(",", ":"),
        ),
        flush=True,
    )


def of_env():
    """OpenFOAM only works after sourcing its bashrc (PATH,
    WM_PROJECT_DIR, LD_LIBRARY_PATH...). Run that once in a
    shell and capture the resulting environment."""

    result = subprocess.run(
        [
            "bash",
            "-c",
            f"source {OF_GLOB} > /dev/null 2>&1 && env",
        ],
        capture_output=True,
        text=True,
        check=True,
    )

    env = os.environ.copy()

    for line in result.stdout.splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            env[key] = value

    return env


def run(env, *cmd, **kwargs):
    result = subprocess.run(
        cmd,
        cwd=CASE_DIR,
        env=env,
        capture_output=True,
        text=True,
        **kwargs,
    )

    if result.returncode != 0:
        print(
            "[of] " + " ".join(cmd) + " failed:",
            file=sys.stderr,
            flush=True,
        )
        print(result.stdout, file=sys.stderr, end="", flush=True)
        print(result.stderr, file=sys.stderr, end="", flush=True)

    return result.returncode


# ------------------------------------------------------------------
# Case files
# ------------------------------------------------------------------

def foam_file_header(obj, cls="dictionary"):
    return (
        "FoamFile\n"
        "{\n"
        "    version 2.0;\n"
        "    format ascii;\n"
        f"    class {cls};\n"
        f"    object {obj};\n"
        "}\n"
        "\n"
    )


def probe_grid(sample_n):
    """Probe locations: a regular sample_n x sample_n lattice at
    the mid-z plane, each point inside a distinct cell."""

    z = THICKNESS / 2.0

    points = []

    for jy in range(sample_n):
        for ix in range(sample_n):
            points.append(
                (
                    (ix + 0.5) / sample_n,
                    (jy + 0.5) / sample_n,
                    z,
                )
            )

    return points


def random_blobs(rng):
    """Random initial hot/cold boxes for this run segment."""

    blobs = []
    count = rng.randint(3, 5)

    for _ in range(count):

        cx = rng.uniform(0.15, 0.85)
        cy = rng.uniform(0.15, 0.85)
        r = rng.uniform(0.08, 0.16)

        temperature = rng.choice([T_MAX, T_MIN])

        blobs.append(
            (
                cx - r,
                cy - r,
                cx + r,
                cy + r,
                temperature,
            )
        )

    return blobs


def write_case(args, blobs):

    shutil.rmtree(CASE_DIR, ignore_errors=True)

    os.makedirs(f"{CASE_DIR}/system")
    os.makedirs(f"{CASE_DIR}/constant")
    os.makedirs(f"{CASE_DIR}/0")

    points = probe_grid(args.sample_n)

    # ---- blockMeshDict: the 2D plate --------------------------

    with open(f"{CASE_DIR}/system/blockMeshDict", "w") as f:
        f.write(foam_file_header("blockMeshDict"))
        f.write(f"""convertToMeters 1;

vertices
(
    (0 0 0)
    (1 0 0)
    (1 1 0)
    (0 1 0)
    (0 0 {THICKNESS})
    (1 0 {THICKNESS})
    (1 1 {THICKNESS})
    (0 1 {THICKNESS})
);

blocks
(
    hex (0 1 2 3 4 5 6 7) ({args.mesh_n} {args.mesh_n} 1)
    simpleGrading (1 1 1)
);

boundary
(
    left
    {{
        type patch;
        faces ((0 4 7 3));
    }}
    right
    {{
        type patch;
        faces ((1 2 6 5));
    }}
    top
    {{
        type patch;
        faces ((2 3 7 6));
    }}
    bottom
    {{
        type patch;
        faces ((0 1 5 4));
    }}
    front
    {{
        type empty;
        faces ((1 0 3 2));
    }}
    back
    {{
        type empty;
        faces ((4 5 6 7));
    }}
);

mergePatchPairs
(
);
""")

    # ---- transportProperties: thermal diffusivity -------------

    with open(f"{CASE_DIR}/constant/transportProperties", "w") as f:
        f.write(foam_file_header("transportProperties"))
        f.write(f"""// thermal diffusivity [m2/s]
DT [0 2 -1 0 0 0 0] {args.alpha};
""")

    # ---- 0/T: hot left, cold right, insulated top/bottom ------

    with open(f"{CASE_DIR}/0/T", "w") as f:
        f.write(foam_file_header("T", "volScalarField"))
        f.write(f"""dimensions [0 0 0 0 1 0 0];

internalField uniform {T_BASE};

boundaryField
{{
    left
    {{
        type fixedValue;
        value uniform {T_MAX};
    }}
    right
    {{
        type fixedValue;
        value uniform {T_MIN};
    }}
    top
    {{
        type zeroGradient;
    }}
    bottom
    {{
        type zeroGradient;
    }}
    front
    {{
        type empty;
    }}
    back
    {{
        type empty;
    }}
}}
""")

    # ---- setFieldsDict: the random blobs ----------------------

    regions = ""

    for x0, y0, x1, y1, temperature in blobs:
        regions += f"""
    boxToCell
    {{
        box ({x0:.4f} {y0:.4f} -1) ({x1:.4f} {y1:.4f} 2);
        fieldValues ( volScalarFieldValue T {temperature:.1f} );
    }}
"""

    with open(f"{CASE_DIR}/system/setFieldsDict", "w") as f:
        f.write(foam_file_header("setFieldsDict"))
        f.write(f"""defaultFieldValues ( volScalarFieldValue T {T_BASE} );

regions
({regions});
""")

    # ---- decomposeParDict --------------------------------------

    with open(f"{CASE_DIR}/system/decomposeParDict", "w") as f:
        f.write(foam_file_header("decomposeParDict"))
        f.write(f"""numberOfSubdomains {args.ranks};

method scotch;
""")

    # ---- fvSchemes / fvSolution --------------------------------

    with open(f"{CASE_DIR}/system/fvSchemes", "w") as f:
        f.write(foam_file_header("fvSchemes"))
        f.write("""ddtSchemes
{
    default Euler;
}

gradSchemes
{
    default Gauss linear;
}

divSchemes
{
    default none;
}

laplacianSchemes
{
    default Gauss linear corrected;
}

interpolationSchemes
{
    default linear;
}

snGradSchemes
{
    default corrected;
}
""")

    with open(f"{CASE_DIR}/system/fvSolution", "w") as f:
        f.write(foam_file_header("fvSolution"))
        f.write("""solvers
{
    T
    {
        solver PCG;
        preconditioner DIC;
        tolerance 1e-7;
        relTol 0.01;
    }
}
""")

    # ---- controlDict (with the probes functionObject) ----------

    probe_lines = "\n".join(
        f"        ({x:.6f} {y:.6f} {z:.6f})"
        for x, y, z in points
    )

    with open(f"{CASE_DIR}/system/controlDict", "w") as f:
        f.write(foam_file_header("controlDict"))
        f.write(f"""application laplacianFoam;

startFrom latestTime;
startTime 0;
stopAt endTime;
endTime {args.steps * args.dt};
deltaT {args.dt};

writeControl none;
writeFrequency {args.stream_every};
purgeWrite 0;
writeFormat ascii;
timeFormat general;
runTimeModifiable false;

functions
{{
    probes
    {{
        type probes;
        libs ("libsampling.so");
        executeControl timeStep;
        executeInterval {args.stream_every};
        writeControl timeStep;
        writeInterval {args.stream_every};
        writeFields false;
        fields (T);
        probeLocations
        (
{probe_lines}
        );
    }}
}}
""")

    return len(points)


# ------------------------------------------------------------------
# Supervising the parallel run
# ------------------------------------------------------------------

class ProbeFile:
    """Tails one rank's postProcessing/probes/0/T file."""

    def __init__(self, path):
        self.path = path
        self.handle = None
        self.buffer = ""
        self.columns = {}      # probe index -> column in rows
        self.next_column = 0
        self.queue = []        # (time_string, [values])

    def poll(self):
        if self.handle is None:
            try:
                self.handle = open(self.path, errors="replace")
            except FileNotFoundError:
                return

            self._read_headers()

        data = self.handle.read()

        if not data:
            return

        self.buffer += data

        while "\n" in self.buffer:
            line, self.buffer = self.buffer.split("\n", 1)

            self._parse_line(line)

    def _read_headers(self):
        """Consume the `# Probe N (x y z)` header lines."""

        while True:
            data = self.handle.read(65536)

            if data:
                self.buffer += data

            if "\n" not in self.buffer:
                return

            line, self.buffer = self.buffer.split("\n", 1)

            self._parse_line(line)

    def _parse_line(self, line):
        line = line.strip()

        if line.startswith("# Probe"):
            # "# Probe 3 (0.25 0.75 0.005)"
            index = int(line.split()[2])
            self.columns[index] = self.next_column
            self.next_column += 1
            return True

        if line.startswith("#"):
            return True

        if not line:
            return True

        parts = line.split()

        self.queue.append(
            (parts[0], [float(v) for v in parts[1:]])
        )

        return True


def normalized(values):
    return [
        round(
            (v - T_MIN) / (T_MAX - T_MIN),
            4,
        )
        for v in values
    ]


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--ranks", type=int, required=True)
    parser.add_argument("--mesh-n", type=int, default=256)
    parser.add_argument("--sample-n", type=int, default=64)
    parser.add_argument("--dt", type=float, default=5.0e-4)
    parser.add_argument("--steps", type=int, default=100000)
    parser.add_argument("--stream-every", type=int, default=50)
    parser.add_argument("--alpha", type=float, default=2.0e-2)

    # Simulated seconds per wall-clock second: the solver is
    # paused whenever it runs ahead of this rate.
    parser.add_argument("--rate", type=float, default=0.25)

    args = parser.parse_args()

    rng = random.Random()

    emit({"type": "status", "message": "Loading OpenFOAM environment..."})
    env = of_env()

    emit({"type": "status", "message": f"Creating {args.mesh_n}x{args.mesh_n} OpenFOAM case..."})

    n_probes = args.sample_n * args.sample_n

    emit({
        "type": "meta",
        "grid_x": args.sample_n,
        "grid_y": args.sample_n,
        "dt": args.dt,
        "mpi_ranks": args.ranks,
    })

    segment = 0
    global_step = 0
    global_time = 0.0

    while True:

        segment += 1

        blobs = random_blobs(rng)
        write_case(args, blobs)

        if run(env, "blockMesh") != 0:
            _fail_segment()

        if run(env, "setFields") != 0:
            _fail_segment()

        emit({
            "type": "status",
            "message": f"Decomposing for {args.ranks} MPI ranks...",
        })

        if run(env, "decomposePar") != 0:
            _fail_segment()

        emit({
            "type": "status",
            "message": "Solving the heat equation...",
        })

        solver = subprocess.Popen(
            [
                "mpirun",
                "--oversubscribe",
                "--bind-to",
                "none",
                "-np",
                str(args.ranks),
                "laplacianFoam",
                "-parallel",
            ],
            cwd=CASE_DIR,
            env=env,

            # OpenFOAM's per-step chatter must not flood the
            # log stream relayed to the browser.
            stdout=subprocess.DEVNULL,
            stderr=None,

            start_new_session=True,
        )

        # In parallel runs the probes functionObject writes
        # one merged file at the case root, with every probe
        # in its own column.
        probes = ProbeFile(
            f"{CASE_DIR}/postProcessing/probes/0/T"
        )

        segment_frames = 0
        wall_start = time.monotonic()
        solver_done = False

        def pace(sim_time):
            """Hold the visual speed constant: pause the solver
            whenever it is ahead of the target rate."""

            target = (
                (time.monotonic() - wall_start) * args.rate
            )

            if sim_time > target + 0.05:
                try:
                    os.killpg(solver.pid, signal.SIGSTOP)
                except ProcessLookupError:
                    return

                while True:
                    target = (
                        (time.monotonic() - wall_start) * args.rate
                    )
                    if sim_time <= target + 0.02:
                        break
                    time.sleep(0.02)

                try:
                    os.killpg(solver.pid, signal.SIGCONT)
                except ProcessLookupError:
                    pass

        while True:

            probes.poll()

            if (
                solver.poll() is not None
                and not probes.queue
            ):
                solver_done = True

            if probes.queue:

                time_key, values = probes.queue.pop(0)

                if len(values) == n_probes:
                    frame = normalized(values)

                    emit({
                        "type": "frame",
                        "step": global_step,
                        "time": global_time,
                        "min": min(frame),
                        "max": max(frame),
                        "values": frame,
                    })

                    segment_frames += 1
                    global_step += args.stream_every
                    global_time += args.dt * args.stream_every

                    pace(float(time_key))

                    continue

            if solver_done:
                break

            time.sleep(0.02)

        return_code = solver.wait()

        if return_code != 0:
            print(
                f"[of] laplacianFoam exited with {return_code}",
                file=sys.stderr,
                flush=True,
            )

            if segment_frames == 0:
                emit({
                    "type": "error",
                    "message": f"OpenFOAM exited with {return_code}.",
                })
                sys.exit(1)

        emit({
            "type": "status",
            "message": "Run complete, restarting with new blobs...",
        })


def _fail_segment():
    emit({
        "type": "error",
        "message": "OpenFOAM case setup failed.",
    })
    sys.exit(1)


if __name__ == "__main__":
    main()
