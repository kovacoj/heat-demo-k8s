import json
import os
import signal
import subprocess
import sys
import threading
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field


BASE_DIR = Path(__file__).resolve().parent
MPI_RANKS = int(os.environ.get("SIM_MPI_RANKS", "16"))

# One 16-core simulation at a time in this pod.
# Later we will replace this with one Kubernetes Job per user.
simulation_lock = threading.Lock()

app = FastAPI(title="Firedrake Heat Demo")


app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://kovacoj.github.io",
        "http://localhost:8000",
    ],
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type"],
)


class SimulationRequest(BaseModel):
    x: float = Field(0.5, ge=0.0, le=1.0)
    y: float = Field(0.5, ge=0.0, le=1.0)

    sigma: float = Field(0.08, ge=0.01, le=0.4)

    diffusivity: float = Field(
        1.0,
        gt=0.0,
        le=10.0,
    )

    dt: float = Field(
        0.0001,
        gt=0.0,
        le=0.01,
    )

    steps: int = Field(
        300,
        ge=1,
        le=2000,
    )

    mesh_n: int = Field(
        1024,
        ge=64,
        le=1024,
    )

    sample_n: int = Field(
        128,
        ge=32,
        le=256,
    )

    stream_every: int = Field(
        5,
        ge=1,
        le=100,
    )


@app.get("/health")
def health():
    return {
        "status": "ok",
        "mpi_ranks": MPI_RANKS,
    }


@app.post("/simulate-stream")
def simulate_stream(req: SimulationRequest):

    # With our current quota we should not launch several
    # independent 16-core MPI runs inside one pod.
    if not simulation_lock.acquire(blocking=False):
        raise HTTPException(
            status_code=409,
            detail="A simulation is already running.",
        )

    process = None

    def stream():

        nonlocal process

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

                "--x",
                str(req.x),

                "--y",
                str(req.y),

                "--sigma",
                str(req.sigma),

                "--diffusivity",
                str(req.diffusivity),

                "--dt",
                str(req.dt),

                "--steps",
                str(req.steps),

                "--mesh-n",
                str(req.mesh_n),

                "--sample-n",
                str(req.sample_n),

                "--stream-every",
                str(req.stream_every),
            ]

            env = os.environ.copy()

            # Prevent every MPI rank from spawning additional
            # OpenMP/BLAS threads.
            env["OMP_NUM_THREADS"] = "1"
            env["OPENBLAS_NUM_THREADS"] = "1"
            env["MKL_NUM_THREADS"] = "1"
            env["PYTHONUNBUFFERED"] = "1"

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

                # Makes it possible to terminate the whole MPI process group.
                start_new_session=True,
            )

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

            for raw_line in process.stdout:

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

            if return_code != 0:
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
            # running, terminate the entire mpiexec process group.
            if process is not None and process.poll() is None:

                try:
                    os.killpg(
                        process.pid,
                        signal.SIGTERM,
                    )

                    process.wait(timeout=5)

                except Exception:

                    try:
                        os.killpg(
                            process.pid,
                            signal.SIGKILL,
                        )
                    except Exception:
                        pass

            simulation_lock.release()

    return StreamingResponse(
        stream(),
        media_type="application/x-ndjson",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "X-Accel-Buffering": "no",
        },
    )
