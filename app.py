import json
import math
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse


BASE_DIR = Path(__file__).resolve().parent
MPI_RANKS = int(os.environ.get("SIM_MPI_RANKS", "16"))

# ------------------------------------------------------------------
# Lava lamp parameters (no user-tunable options).
# ------------------------------------------------------------------

# FEM mesh: 32 x 64 cells over a 1 x 2 domain.
MESH_N = 32

# Sampling grid sent to the browser: 20 x 40 points.
SAMPLE_N = 20

# Rayleigh number: buoyancy vs. viscosity — moderate, so
# blobs drift lazily instead of being shredded by turbulence.
RA = 10000.0

# Prandtl number: momentum diffuses much faster than heat,
# like very viscous wax in a lamp — smooth velocity field,
# coherent blobs.
PR = 10.0

# Bottom plate temperature (top and walls are 0). The
# effective thermal forcing scales with RA * T_HOT.
T_HOT = 2.0

# Computational slow motion. Visual speed is steps/second
# times dt, and dt is bounded by the CFL limit — so the only
# way to slow the lamp is to step below that limit.
PLAYBACK = 40.0

# Explicit advection is CFL-limited: peak |u| ~ sqrt(RA * T_HOT).
DT = 0.7 / (MESH_N * math.sqrt(RA * T_HOT)) / PLAYBACK

# One frame every N timesteps (~15 frames/s on 16 ranks).
STREAM_EVERY = 10

# Effectively endless: the run ends when the browser
# disconnects (or Stop is pressed), or on divergence.
STEPS = 10**6

# If the MPI job produces no output for this long
# (broken ranks spin at 100% CPU without any progress),
# it is considered stalled and killed.
STALL_TIMEOUT = 120.0

# After the worker process dies, wait this long for the
# response generator to finish before force-releasing
# the lock. If the browser disappeared while the worker
# was stalled, the generator stays suspended at a yield
# forever and its `finally` never runs.
DEAD_GRACE = 30.0

# ------------------------------------------------------------------

# One 16-core simulation at a time in this pod.
simulation_lock = threading.Lock()

# Serializes lock acquire/release and identifies the
# current owner, so a delayed cleanup from an old request
# can never release a lock that a newer request holds.
_lock_guard = threading.Lock()

_lock_owner = {"token": None}

# State dict of the request currently holding the lock.
_active = {"state": None}

app = FastAPI(title="Firedrake Lava Lamp")


app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://kovacoj.github.io",
        "http://localhost:8000",
    ],
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type"],
)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "mpi_ranks": MPI_RANKS,
    }


@app.post("/simulate-stream")
def simulate_stream():

    # Unique token for this request.
    token = object()

    # With our current quota we should not launch several
    # independent 16-core MPI runs inside one pod. A new
    # request (page reload) takes over: it kills the running
    # job and waits briefly for the old request's cleanup.
    acquired = False
    victim = None

    with _lock_guard:

        if simulation_lock.acquire(blocking=False):
            acquired = True
        else:
            victim = _active["state"]

    if acquired:
        _lock_owner["token"] = token
    else:
        if victim is not None:
            victim["takeover"] = True
            victim["kill_session"](signal.SIGKILL)

        deadline = time.monotonic() + 15.0

        while time.monotonic() < deadline:

            with _lock_guard:

                if simulation_lock.acquire(blocking=False):
                    acquired = True
                    _lock_owner["token"] = token
                    break

                # Someone else took over before us.
                if _active["state"] is not victim:
                    break

            time.sleep(0.25)

        if not acquired:
            raise HTTPException(
                status_code=409,
                detail="A simulation is already running.",
            )

    # Per-request state, shared between the response
    # generator and the watchdog thread.
    state = {
        "process": None,
        "last_output": None,
        "stalled": False,
        "released": False,
        "takeover": False,
    }

    with _lock_guard:
        _active["state"] = state

    def release_lock():
        # Exactly-once release, and it must never release
        # a lock that a newer request has re-acquired.
        with _lock_guard:

            if state["released"]:
                return

            state["released"] = True

            if _lock_owner["token"] is token:

                try:
                    simulation_lock.release()
                except RuntimeError:
                    pass

    def kill_session(sig):
        process = state["process"]
        if process is None:
            return

        # mpiexec and its own process group.
        try:
            os.killpg(process.pid, sig)
        except Exception:
            pass

        # The MPI ranks get their own process group,
        # but they share mpiexec's session.
        try:
            subprocess.run(
                ["pkill", f"-{int(sig)}", "-s", str(process.pid)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except Exception:
            pass

    state["kill_session"] = kill_session

    def watchdog():

        # Wait for the process to be spawned.
        for _ in range(50):
            if state["process"] is not None:
                break
            time.sleep(0.2)

        process = state["process"]

        if process is None:
            release_lock()
            return

        while process.poll() is None:
            time.sleep(1)

            last_output = state["last_output"]

            if (
                last_output is not None
                and time.monotonic() - last_output > STALL_TIMEOUT
            ):
                state["stalled"] = True

                print(
                    "[watchdog] No worker output for "
                    f"{STALL_TIMEOUT:.0f}s, killing MPI session.",
                    file=sys.stderr,
                    flush=True,
                )

                kill_session(signal.SIGKILL)

                break

        # The process is gone (finished or killed). Give the
        # response generator time to run its cleanup, then
        # make sure the lock is released even if the generator
        # was abandoned while blocked on a pipe read. A new
        # request demanding takeover releases immediately.
        deadline = time.monotonic() + DEAD_GRACE

        while time.monotonic() < deadline:

            if state["takeover"]:
                break

            time.sleep(0.5)

        release_lock()

    def stream():

        try:

            # Let the browser know immediately that something is happening.
            yield json.dumps({
                "type": "status",
                "message": f"Starting {MPI_RANKS}-rank MPI simulation..."
            }) + "\n"

            cmd = [
                "mpiexec",
                "-n",
                str(MPI_RANKS),

                "python3",
                "-u",
                str(BASE_DIR / "worker.py"),

                "--ra",
                str(RA),

                "--pr",
                str(PR),

                "--t-hot",
                str(T_HOT),

                "--mesh-n",
                str(MESH_N),

                "--sample-n",
                str(SAMPLE_N),

                "--dt",
                str(DT),

                "--steps",
                str(STEPS),

                "--stream-every",
                str(STREAM_EVERY),
            ]

            env = os.environ.copy()

            # Prevent every MPI rank from spawning additional
            # OpenMP/BLAS threads.
            env["OMP_NUM_THREADS"] = "1"
            env["OPENBLAS_NUM_THREADS"] = "1"
            env["MKL_NUM_THREADS"] = "1"
            env["PYTHONUNBUFFERED"] = "1"

            state["last_output"] = time.monotonic()

            process = subprocess.Popen(
                cmd,
                cwd=str(BASE_DIR),
                env=env,

                stdout=subprocess.PIPE,

                # Worker stderr MUST NOT share the stdout pipe:
                # MPICH diagnostics from all ranks would interleave
                # with rank 0's long NDJSON frame lines and corrupt
                # the protocol stream. Drain it on a separate pipe.
                stderr=subprocess.PIPE,

                text=True,
                bufsize=1,

                # Makes it possible to terminate the whole MPI
                # process tree via the session id.
                start_new_session=True,
            )

            state["process"] = process

            assert process.stdout is not None
            assert process.stderr is not None

            def drain_stderr():

                for raw_line in process.stderr:

                    # Firedrake/PETSc/MPI diagnostics go to Kubernetes logs,
                    # not to the browser protocol.
                    print(
                        "[worker]",
                        raw_line,
                        end="",
                        file=sys.stderr,
                        flush=True,
                    )

            # Without draining, the stderr pipe would fill up (64 KiB)
            # and block the whole MPI job.
            stderr_thread = threading.Thread(
                target=drain_stderr,
                daemon=True,
            )

            stderr_thread.start()

            watchdog_thread = threading.Thread(
                target=watchdog,
                daemon=True,
            )

            watchdog_thread.start()

            for raw_line in process.stdout:

                state["last_output"] = time.monotonic()

                if raw_line.startswith("NDJSON:"):

                    # Strip our protocol prefix.
                    yield raw_line[len("NDJSON:"):]

                else:

                    # PETSc stdout warnings etc. go to Kubernetes logs.
                    print(
                        "[worker]",
                        raw_line,
                        end="",
                        file=sys.stderr,
                        flush=True,
                    )

            return_code = process.wait()

            stderr_thread.join(timeout=5)

            if state["stalled"]:
                yield json.dumps({
                    "type": "error",
                    "message": (
                        "The MPI job stalled (no output for "
                        f"{STALL_TIMEOUT:.0f}s) and was killed. "
                        "Please try again."
                    ),
                }) + "\n"

            elif return_code != 0:
                yield json.dumps({
                    "type": "error",
                    "message": (
                        f"Firedrake worker exited with code "
                        f"{return_code}"
                    ),
                }) + "\n"

        except GeneratorExit:
            # Browser disconnected / Stop pressed.
            raise

        except Exception as exc:

            print(
                f"Simulation controller error: {exc}",
                file=sys.stderr,
                flush=True,
            )

            yield json.dumps({
                "type": "error",
                "message": str(exc),
            }) + "\n"

        finally:

            # If the browser disappeared while the MPI job was still
            # running, terminate the entire process tree.
            process = state["process"]

            if process is not None and process.poll() is None:

                kill_session(signal.SIGTERM)

                try:
                    process.wait(timeout=5)
                except Exception:
                    kill_session(signal.SIGKILL)

            release_lock()

    return StreamingResponse(
        stream(),
        media_type="application/x-ndjson",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "X-Accel-Buffering": "no",
        },
    )
