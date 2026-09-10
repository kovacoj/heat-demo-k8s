"""FastAPI controller for the weak-form heat demo (spec section 18).

Public endpoints:
    GET  /api/health
    POST /api/nn/predict           one NN forward pass, 26 frames
    POST /api/firedrake/run        creates a Kubernetes Job, returns at once
    GET  /api/firedrake/{job_id}   pending | running | done | error
    GET  /api/metrics              static, measured evaluation metadata

Internal (X-Callback-Token):
    POST /internal/result/{job_id}
"""
import os
import threading
import time

import numpy as np
import scipy.sparse as sp
from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware

from api.inference import InferenceEngine
from api.jobs import JobManager, MAX_CONCURRENT
from api.schemas import HeatParams, JobStatus, NNResponse, RunResponse

CALLBACK_TOKEN = os.environ.get("CALLBACK_TOKEN", "")
OPERATORS_DIR = os.environ.get("WEAKHEAT_OPERATORS", "/app/data/operators")
RESULT_TTL_S = 1800          # forget finished results after 30 min
STALE_JOB_S = 900            # no callback within 15 min -> error

app = FastAPI(title="weakheat-demo", docs_url=None, redoc_url=None)

pages_origin = os.environ.get("PAGES_ORIGIN", "")
allowed = ["http://localhost:8080", "http://127.0.0.1:8080"]
if pages_origin:
    allowed.append(pages_origin)
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization"],
)

engine = InferenceEngine()
jobs = JobManager()

_results: dict = {}          # job_id -> dict(status=..., params=..., created=...)
_lock = threading.Lock()

M = sp.load_npz(os.path.join(OPERATORS_DIR, "M.npz"))
order = np.load(os.path.join(OPERATORS_DIR, "order.npy"))


@app.get("/api/health")
def health():
    return {"status": "ok", "model_parameters": engine.n_params}


@app.post("/api/nn/predict", response_model=NNResponse)
def nn_predict(p: HeatParams):
    out = engine.predict(p.x0, p.y0, p.sigma, p.alpha)
    return out


@app.post("/api/firedrake/run", response_model=RunResponse)
def firedrake_run(p: HeatParams):
    with _lock:
        active = jobs.active_jobs()
        if active >= MAX_CONCURRENT:
            raise HTTPException(
                status_code=429,
                detail=f"{active} Firedrake jobs already running, try again shortly")
        job_id = jobs.new_job_id()
        try:
            jobs.create_job(job_id, p.model_dump(), CALLBACK_TOKEN)
        except Exception as e:  # noqa: BLE001
            raise HTTPException(status_code=503, detail=f"job creation failed: {e}")
        _results[job_id] = {
            "status": "pending",
            "params": p.model_dump(),
            "created": time.time(),
        }
    return RunResponse(job_id=job_id, status="pending")


@app.get("/api/firedrake/{job_id}", response_model=JobStatus)
def firedrake_status(job_id: str):
    with _lock:
        rec = _results.get(job_id)
    if rec is None:
        raise HTTPException(status_code=404, detail="unknown job_id")
    age = time.time() - rec["created"]
    if rec["status"] in ("pending", "running"):
        if age > STALE_JOB_S:
            rec["status"] = "error"
            rec["error"] = "job timed out"
        else:
            phase = jobs.job_phase(job_id)
            if phase == "running" and rec["status"] == "pending":
                rec["status"] = "running"
    out = {"job_id": job_id, "status": rec["status"]}
    for key in ("times", "frames", "runtime_ms", "shape", "error", "relative_l2"):
        if rec.get(key) is not None:
            out[key] = rec[key]
    return out


@app.post("/internal/result/{job_id}")
async def internal_result(job_id: str, request: Request,
                          x_callback_token: str = Header(default="")):
    if not CALLBACK_TOKEN or x_callback_token != CALLBACK_TOKEN:
        raise HTTPException(status_code=403, detail="invalid callback token")
    payload = await request.json()
    with _lock:
        rec = _results.setdefault(
            job_id, {"params": None, "created": time.time()})
        if payload.get("status") == "done":
            try:
                u_fd = np.asarray(payload["frames"], dtype=np.float64)  # [T, N]
                if u_fd.ndim != 2 or u_fd.shape[1] != len(order):
                    raise ValueError("malformed frames")
                u_fd = u_fd[:, np.argsort(order)]  # back to native dof order
                rel_l2 = None
                if rec.get("params"):
                    q = rec["params"]
                    nn = engine.predict(q["x0"], q["y0"], q["sigma"], q["alpha"])
                    u_nn = np.asarray(nn["frames"], dtype=np.float64)[:, np.argsort(order)]
                    e = u_nn - u_fd
                    num = float(np.einsum("tn,nm,tm->t", e, M.toarray(), e).sum())
                    den = float(np.einsum("tn,nm,tm->t", u_fd, M.toarray(), u_fd).sum())
                    if den > 1e-12:
                        rel_l2 = float(np.sqrt(num / den))
                rec.update(status="done", times=payload["times"],
                           frames=payload["frames"],
                           runtime_ms=payload.get("runtime_ms"),
                           shape=payload.get("shape"), relative_l2=rel_l2)
            except Exception as e:  # noqa: BLE001
                rec.update(status="error", error=f"invalid result payload: {e}")
        else:
            rec.update(status="error",
                       error=str(payload.get("error", "unknown error"))[:300])
    # opportunistic cleanup of old records
    now = time.time()
    with _lock:
        stale = [k for k, v in _results.items()
                 if now - v.get("created", now) > RESULT_TTL_S]
        for k in stale:
            _results.pop(k, None)
    return {"ok": True}


@app.get("/api/metrics")
def metrics():
    m = dict(engine.metrics)
    m.setdefault("pde", "2D heat equation")
    m.setdefault("mesh", "32x32 CG1")
    m.setdefault("ndofs", 1089)
    m["model_parameters"] = engine.n_params
    return m
