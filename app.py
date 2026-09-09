import json
import math
import os
import sys
import threading
import time
import uuid
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from kubernetes import client as k8s
from kubernetes import config as k8s_config


BASE_DIR = Path(__file__).resolve().parent

# ------------------------------------------------------------------
# Lava lamp parameters (no user-tunable options).
# ------------------------------------------------------------------

# FEM mesh: 64 x 128 cells over a 1 x 2 domain.
MESH_N = 64

# Sampling grid sent to the browser: 64 x 128 points.
SAMPLE_N = 64

# Rayleigh number: buoyancy vs. viscosity — moderate, so
# blobs drift lazily instead of being shredded by turbulence.
RA = 10000.0

# Prandtl number: momentum diffuses much faster than heat,
# like very viscous wax in a lamp — smooth velocity field,
# coherent blobs.
PR = 10.0

# Bottom plate temperature (top and walls are 0). The
# effective thermal forcing scales with RA * T_HOT.
T_HOT = 2.5

# Computational slow motion. Visual speed is steps/second
# times dt, and dt is bounded by the CFL limit — so the only
# way to slow the lamp is to step below that limit. Playback 1
# (dt at the CFL limit) is numerically unstable: the explicit
# buoyancy source in the vorticity equation exceeds its
# stability bound and the worker blows up right after frame 0.
PLAYBACK = 2.0

# Explicit advection is CFL-limited: peak |u| ~ sqrt(RA * T_HOT).
DT = 0.7 / (MESH_N * math.sqrt(RA * T_HOT)) / PLAYBACK

# One frame every N timesteps (~8 frames/s on 10 ranks).
STREAM_EVERY = 10

# Wax phase field (Cahn-Hilliard): interface width and
# mobility. dt < eps² / (M · max|f''|) keeps the linearized
# scheme stable: 0.06² / (4 · 2) ≈ 4.5e-4 >> DT. Mobility
# is low on purpose: the flow (shear >> surface tension)
# shreds the wax faster than CH can re-merge it, so the
# lamp never settles into a single dead blob.
CH_EPSILON = 0.06
CH_MOBILITY = 4.0

# Extra thermal expansion of the wax: hot wax is more buoyant
# than the hot fluid around it and rises, cools near the top
# and sinks — the lava-lamp cycle. The term is explicit like
# the base buoyancy, so it must stay small enough for
# stability (the c·T gradient is steep at the interfaces).
WAX_BUOYANCY = 1.0

# Effectively endless: the run ends when the browser
# disconnects, or on divergence.
STEPS = 10**6

# ------------------------------------------------------------------
# Cahn-Hilliard phase separation (from the ../chns project).
# ------------------------------------------------------------------

# FEM mesh: 64 x 64 cells over the unit square.
CH_MESH_N = 64

# Sampling grid sent to the browser: 64 x 64 points.
CH_SAMPLE_N = 64

# Coarsening is slow; larger dt keeps the demo lively
# (the scheme is Crank-Nicolson and tolerates it).
CH_DT = 1.0e-2

CH_STREAM_EVERY = 1

# ------------------------------------------------------------------
# Cahn-Hilliard-Navier-Stokes rising bubbles (../chns project).
# ------------------------------------------------------------------

# FEM mesh: 64 x 192 cells over a 1 x 3 box.
CHNS_MESH_N = 64

# Sampling grid sent to the browser: 64 x 192 points.
CHNS_SAMPLE_N = 64

CHNS_DT = 1.0e-3

CHNS_STREAM_EVERY = 1

# ------------------------------------------------------------------
# OpenFOAM heat equation (laplacianFoam, its minimal solver).
# ------------------------------------------------------------------

# FVM mesh: 256 x 256 cells on the unit square (one cell thick).
OF_MESH_N = 256

# Sampling grid sent to the browser: 64 x 64 probe points.
OF_SAMPLE_N = 64

# Thermal diffusivity [m2/s]: fills the plate in ~50 sim-s.
OF_ALPHA = 2.0e-2

OF_DT = 5.0e-4

# One segment of the endless restart loop.
OF_STEPS = 100000

# Probes are written every N timesteps.
OF_STREAM_EVERY = 50

# Simulated seconds per wall-clock second; the solver is
# paused whenever it runs ahead of this pace.
OF_RATE = 0.25

# ------------------------------------------------------------------
# Kubernetes job launcher
# ------------------------------------------------------------------

# Images the simulation jobs run: the same Firedrake image as
# this API pod for the PDE solvers, and a separate OpenFOAM
# image. Injected by the Makefile on deploy.
SIM_IMAGE = os.environ.get("SIM_IMAGE", "")
OF_IMAGE = os.environ.get("OF_IMAGE", "")

MPI_RANKS = int(os.environ.get("SIM_MPI_RANKS", "10"))

# How many browsers can run a simulation at the same time.
# Jobs request 6 CPUs and burst to a 10-CPU limit; the API
# pod requests 200m / limits 1. The namespace quota is
# 20 requested / 32 limited CPUs, so three jobs fit with
# room to spare on requests and exactly one spare limit —
# a fourth job is rejected by the limits quota (and by us).
# The simulation is a continuously busy MPI workload, so
# the request is the guaranteed floor under contention,
# not a reservation for idle periods.
MAX_SIMS = int(os.environ.get("SIM_MAX_SIMS", "3"))

JOB_CPU_REQUEST = "6"
JOB_CPU_LIMIT = "10"
JOB_MEMORY_REQUEST = "6Gi"
JOB_MEMORY_LIMIT = "10Gi"

# A fresh job may stay Pending this long before we give up
# (scheduling plus a cold image pull can take a while).
PENDING_TIMEOUT = 120.0

# Hard cap on one simulation, and the backstop for orphaned
# jobs if this API pod dies with browsers attached.
ACTIVE_DEADLINE = 1800

# Finished job objects are removed automatically.
TTL_AFTER_FINISH = 60

# If a running job produces no log output for this long, it
# is stalled (broken MPI ranks spin at 100% CPU without any
# progress) and is deleted.
STALL_TIMEOUT = 120.0

# The MPI runtime on this cluster is intermittently flaky
# during startup (ranks die with "Read -1, errno = 1").
# A job that dies before producing a single frame is retried.
MAX_ATTEMPTS = 3

try:
    k8s_config.load_incluster_config()

    with open(
        "/var/run/secrets/kubernetes.io/serviceaccount/namespace"
    ) as f:
        NAMESPACE = f.read().strip()

    batch_api = k8s.BatchV1Api()
    core_api = k8s.CoreV1Api()
    K8S_ERROR = None
except Exception as exc:
    NAMESPACE = None
    batch_api = None
    core_api = None
    K8S_ERROR = str(exc)

# Serializes the free-slot check + job creation, so two
# simultaneous requests cannot both grab the last slot.
_slots_guard = threading.Lock()


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


@app.on_event("startup")
def sweep_old_jobs():
    """This API pod just started, so no browser can be attached
    to a simulation yet: any leftover simulation jobs are
    orphans from a previous incarnation and are deleted."""

    if batch_api is None:
        return

    jobs = batch_api.list_namespaced_job(
        namespace=NAMESPACE,
        label_selector="app=heat-simulation",
    )

    for job in jobs.items:
        delete_job(job.metadata.name)


def ndjson(message):
    return json.dumps(message, separators=(",", ":")) + "\n"


def delete_job(name):
    try:
        batch_api.delete_namespaced_job(
            name,
            namespace=NAMESPACE,
            propagation_policy="Background",
        )
    except Exception:
        pass


def active_sim_count():
    jobs = batch_api.list_namespaced_job(
        namespace=NAMESPACE,
        label_selector="app=heat-simulation",
    )

    count = 0

    for job in jobs.items:
        status = job.status

        if status is None or not (status.succeeded or status.failed):
            count += 1

    return count


def make_job_body(name, cmd, image):
    return {
        "apiVersion": "batch/v1",
        "kind": "Job",
        "metadata": {
            "name": name,
            "labels": {
                "app": "heat-simulation",
                "sim": name,
            },
        },
        "spec": {
            "backoffLimit": 0,
            "activeDeadlineSeconds": ACTIVE_DEADLINE,
            "ttlSecondsAfterFinished": TTL_AFTER_FINISH,
            # No manual spec.selector: this cluster rejects
            # non-auto-generated ones (422). Kubernetes derives
            # a unique controller-uid selector on its own; the
            # template labels below remain ours to list/delete
            # jobs by.
            "template": {
                "metadata": {
                    "labels": {
                        "app": "heat-simulation",
                        "sim": name,
                    },
                },
                "spec": {
                    "restartPolicy": "Never",
                    "terminationGracePeriodSeconds": 10,
                    "securityContext": {
                        "runAsUser": 1000,
                        "runAsNonRoot": True,
                        "seccompProfile": {
                            "type": "RuntimeDefault",
                        },
                    },
                    "containers": [
                        {
                            "name": "worker",
                            "image": image,
                            "imagePullPolicy": "Always",
                            "command": cmd,
                            "env": [
                                {
                                    "name": "XDG_CACHE_HOME",
                                    "value": "/tmp/.cache",
                                },
                            ],
                            "securityContext": {
                                "allowPrivilegeEscalation": False,
                                "capabilities": {
                                    "drop": ["ALL"],
                                },
                            },
                            "resources": {
                                "requests": {
                                    "cpu": JOB_CPU_REQUEST,
                                    "memory": JOB_MEMORY_REQUEST,
                                },
                                "limits": {
                                    "cpu": JOB_CPU_LIMIT,
                                    "memory": JOB_MEMORY_LIMIT,
                                },
                            },
                            "volumeMounts": [
                                {
                                    "name": "dshm",
                                    "mountPath": "/dev/shm",
                                },
                            ],
                        },
                    ],
                    "volumes": [
                        {
                            "name": "dshm",
                            "emptyDir": {
                                "medium": "Memory",
                                "sizeLimit": "1Gi",
                            },
                        },
                    ],
                },
            },
        },
    }


@app.get("/health")
def health():
    try:
        slots_free = max(0, MAX_SIMS - active_sim_count())
    except Exception:
        # The Kubernetes API being unreachable must not flap
        # this pod's probes.
        slots_free = MAX_SIMS

    return {
        "status": "ok",
        "mpi_ranks": MPI_RANKS,
        "slots_free": slots_free,
    }


def build_lamp_cmd():
    return [
        "mpiexec",
        "-n",
        str(MPI_RANKS),
        "python3",
        "-u",
        str(BASE_DIR / "worker.py"),
        "--ra", str(RA),
        "--pr", str(PR),
        "--t-hot", str(T_HOT),
        "--mesh-n", str(MESH_N),
        "--sample-n", str(SAMPLE_N),
        "--dt", str(DT),
        "--steps", str(STEPS),
        "--stream-every", str(STREAM_EVERY),
        "--ch-epsilon", str(CH_EPSILON),
        "--ch-mobility", str(CH_MOBILITY),
        "--wax-buoyancy", str(WAX_BUOYANCY),
    ]


def build_ch_cmd():
    return [
        "mpiexec",
        "-n",
        str(MPI_RANKS),
        "python3",
        "-u",
        str(BASE_DIR / "worker_ch.py"),
        "--mesh-n", str(CH_MESH_N),
        "--sample-n", str(CH_SAMPLE_N),
        "--dt", str(CH_DT),
        "--steps", str(STEPS),
        "--stream-every", str(CH_STREAM_EVERY),
    ]


def build_chns_cmd():
    return [
        "mpiexec",
        "-n",
        str(MPI_RANKS),
        "python3",
        "-u",
        str(BASE_DIR / "worker_chns.py"),
        "--mesh-n", str(CHNS_MESH_N),
        "--sample-n", str(CHNS_SAMPLE_N),
        "--dt", str(CHNS_DT),
        "--steps", str(STEPS),
        "--stream-every", str(CHNS_STREAM_EVERY),
    ]


def build_of_cmd():
    return [
        "python3",
        "-u",
        "/opt/worker_of.py",
        "--ranks", str(MPI_RANKS),
        "--mesh-n", str(OF_MESH_N),
        "--sample-n", str(OF_SAMPLE_N),
        "--dt", str(OF_DT),
        "--steps", str(OF_STEPS),
        "--stream-every", str(OF_STREAM_EVERY),
        "--alpha", str(OF_ALPHA),
        "--rate", str(OF_RATE),
    ]


def simulation_endpoint(build_cmd, image):
    """Each browser session gets its own Kubernetes Job: a
    dedicated 10-CPU pod running the MPI worker. Frames are
    relayed by following the pod's logs. The job is deleted
    when the browser disconnects."""

    def simulate():
        return StreamingResponse(
            stream(),
            media_type="application/x-ndjson",
            headers={
                "Cache-Control": "no-cache, no-transform",
                "X-Accel-Buffering": "no",
            },
        )

    def stream():

        # Shared with the watchdog thread.
        state = {
            "last_output": None,
            "stalled": False,
            "done": False,
            "job": None,
        }

        frames = 0
        saw_done = False
        resp = None

        def watchdog():

            # Wait until a job exists.
            while state["job"] is None and not state["done"]:
                time.sleep(0.2)

            while not state["done"]:

                time.sleep(2.0)

                last_output = state["last_output"]

                if (
                    last_output is not None
                    and time.monotonic() - last_output > STALL_TIMEOUT
                ):

                    state["stalled"] = True

                    print(
                        f"[watchdog] No worker output for "
                        f"{STALL_TIMEOUT:.0f}s, deleting job "
                        f"{state['job']}.",
                        file=sys.stderr,
                        flush=True,
                    )

                    delete_job(state["job"])
                    return

        try:
            if K8S_ERROR is not None:
                yield ndjson({
                    "type": "error",
                    "message": f"Kubernetes API unavailable: {K8S_ERROR}",
                })
                return

            if not image:
                yield ndjson({
                    "type": "error",
                    "message": "Simulation image is not configured.",
                })
                return

            yield ndjson({
                "type": "status",
                "message": (
                    f"Starting {MPI_RANKS}-rank MPI simulation pod..."
                ),
            })

            cmd = build_cmd()

            for attempt in range(MAX_ATTEMPTS):

                with _slots_guard:

                    busy = active_sim_count() >= MAX_SIMS

                    if not busy:
                        name = f"heat-sim-{uuid.uuid4().hex[:8]}"
                        batch_api.create_namespaced_job(
                            NAMESPACE,
                            make_job_body(name, cmd, image),
                        )
                        state["job"] = name
                        state["last_output"] = None

                if busy:
                    yield ndjson({
                        "type": "error",
                        "message": (
                            "All simulation slots are busy — "
                            "try again in a moment."
                        ),
                    })
                    return

                # Wait for the job's pod to start running.
                # Status lines keep flowing so a disconnecting
                # browser is noticed even while waiting.
                deadline = time.monotonic() + PENDING_TIMEOUT
                next_status = 0.0
                pod = None

                while True:

                    pods = core_api.list_namespaced_pod(
                        namespace=NAMESPACE,
                        label_selector=f"sim={name}",
                    ).items

                    phase = pods[0].status.phase if pods else None

                    if phase in ("Running", "Failed", "Succeeded"):
                        pod = pods[0].metadata.name
                        break

                    if time.monotonic() > deadline:
                        break

                    if time.monotonic() > next_status:
                        yield ndjson({
                            "type": "status",
                            "message": (
                                "Scheduling simulation pod (a cold "
                                "image pull can take a minute)..."
                            ),
                        })
                        next_status = time.monotonic() + 5.0

                    time.sleep(0.5)

                if pod is None:
                    delete_job(name)
                    state["job"] = None

                    yield ndjson({
                        "type": "error",
                        "message": (
                            "No simulation slot became available — "
                            "try again in a moment."
                        ),
                    })
                    return

                if phase != "Running":
                    # The pod died before starting to stream
                    # (MPICH startup flake): delete and retry.
                    delete_job(name)
                    state["job"] = None

                    print(
                        f"[retry] Pod phase {phase} before any "
                        f"frame (attempt {attempt + 1}/"
                        f"{MAX_ATTEMPTS}).",
                        file=sys.stderr,
                        flush=True,
                    )

                    time.sleep(1.0)
                    continue

                # Follow the pod's logs and relay the NDJSON.
                state["last_output"] = time.monotonic()

                if attempt == 0:
                    threading.Thread(
                        target=watchdog,
                        daemon=True,
                    ).start()

                try:
                    resp = core_api.read_namespaced_pod_log(
                        pod,
                        NAMESPACE,
                        follow=True,
                        _preload_content=False,
                    )

                    for raw_line in resp:

                        state["last_output"] = time.monotonic()

                        line = raw_line.decode(
                            "utf-8",
                            errors="replace",
                        )

                        if line.startswith("NDJSON:"):

                            if '"type":"frame"' in line:
                                frames += 1

                            if '"type":"done"' in line:
                                saw_done = True

                            yield line[len("NDJSON:"):]

                        else:
                            # PETSc stdout warnings etc. go to
                            # this API pod's logs.
                            print(
                                "[worker]",
                                line,
                                end="",
                                file=sys.stderr,
                                flush=True,
                            )

                except Exception as exc:
                    print(
                        f"[sim] Log stream error: {exc}",
                        file=sys.stderr,
                        flush=True,
                    )

                finally:
                    if resp is not None:
                        try:
                            resp.close()
                        except Exception:
                            pass
                        resp = None

                if frames > 0:

                    if state["stalled"]:
                        yield ndjson({
                            "type": "error",
                            "message": (
                                "The MPI job stalled (no output for "
                                f"{STALL_TIMEOUT:.0f}s) and was "
                                "killed. Please try again."
                            ),
                        })

                    elif not saw_done:
                        yield ndjson({
                            "type": "error",
                            "message": "Simulation exited unexpectedly.",
                        })

                    return

                # The job died before producing a single frame
                # (the MPI runtime on this cluster is
                # intermittently flaky during startup).
                delete_job(name)
                state["job"] = None

                print(
                    f"[retry] Worker died before any frame "
                    f"(attempt {attempt + 1}/{MAX_ATTEMPTS}).",
                    file=sys.stderr,
                    flush=True,
                )

                time.sleep(1.0)

            yield ndjson({
                "type": "error",
                "message": (
                    "The simulation kept dying during startup. "
                    "Please try again."
                ),
            })

        except GeneratorExit:
            # Browser disconnected / Stop pressed.
            raise

        except Exception as exc:

            import traceback

            traceback.print_exc()

            yield ndjson({
                "type": "error",
                "message": str(exc),
            })

        finally:

            state["done"] = True

            if resp is not None:
                try:
                    resp.close()
                except Exception:
                    pass

            # The job (and its pod) must not outlive the
            # browser session.
            if state["job"] is not None:
                delete_job(state["job"])

    return simulate


app.post("/simulate-stream")(simulation_endpoint(build_lamp_cmd, SIM_IMAGE))
app.post("/simulate-ch")(simulation_endpoint(build_ch_cmd, SIM_IMAGE))
app.post("/simulate-chns")(simulation_endpoint(build_chns_cmd, SIM_IMAGE))
app.post("/simulate-of")(simulation_endpoint(build_of_cmd, OF_IMAGE))
