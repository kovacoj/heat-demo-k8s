"""API tests with FastAPI TestClient (no cluster, dummy model)."""
import os
import sys

import numpy as np
import pytest

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, ROOT)

os.environ.setdefault("DEMO_TOKEN", "unit-test-token")
os.environ.setdefault("CALLBACK_TOKEN", "unit-test-callback")
os.environ.setdefault(
    "WEAKHEAT_CKPT", os.path.join(ROOT, "checkpoints", "localtest", "weakheat_best.pt"))
os.environ.setdefault("WEAKHEAT_META",
                      os.path.join(ROOT, "checkpoints", "localtest", "model_meta.json"))
os.environ.setdefault("WEAKHEAT_METRICS",
                      os.path.join(ROOT, "presentation", "metrics.json"))
os.environ.setdefault("WEAKHEAT_OPERATORS",
                      os.path.join(ROOT, "data", "operators"))

from fastapi.testclient import TestClient  # noqa: E402

import api.main as main  # noqa: E402

# no cluster in unit tests
main.jobs.available = False


@pytest.fixture(scope="module")
def client():
    with TestClient(main.app) as c:
        yield c


AUTH = {"Authorization": "Bearer unit-test-token"}


def test_api_parameter_validation(client):
    for bad in ({"x0": 0.9, "y0": 0.5, "sigma": 0.07, "alpha": 0.01},
                {"x0": 0.5, "y0": 0.05, "sigma": 0.07, "alpha": 0.01},
                {"x0": 0.5, "y0": 0.5, "sigma": 0.2, "alpha": 0.01},
                {"x0": 0.5, "y0": 0.5, "sigma": 0.07, "alpha": 0.1}):
        r = client.post("/api/nn/predict", json=bad, headers=AUTH)
        assert r.status_code == 422, bad


def test_nn_endpoint(client):
    r = client.post("/api/nn/predict",
                   json={"x0": 0.4, "y0": 0.6, "sigma": 0.07, "alpha": 0.01},
                   headers=AUTH)
    assert r.status_code == 200
    body = r.json()
    assert len(body["times"]) == 26
    assert len(body["frames"]) == 26
    assert len(body["frames"][0]) == 33 * 33
    assert body["inference_ms"] >= 0
    assert np.isfinite(np.asarray(body["frames"])).all()


def test_auth_required(client):
    r = client.post("/api/nn/predict",
                    json={"x0": 0.4, "y0": 0.6, "sigma": 0.07, "alpha": 0.01})
    assert r.status_code == 401


def test_firedrake_run_without_cluster_503(client):
    r = client.post("/api/firedrake/run",
                    json={"x0": 0.4, "y0": 0.6, "sigma": 0.07, "alpha": 0.01},
                    headers=AUTH)
    assert r.status_code == 503


def test_callback_auth_and_flow(client):
    rng = np.random.default_rng(1)
    frames = np.abs(rng.normal(0.3, 0.1, (26, 1089))).tolist()
    payload = {"job_id": "unit01", "status": "done",
               "times": [round(i * 0.01, 2) for i in range(26)],
               "frames": frames, "runtime_ms": 1234.0, "shape": [33, 33]}
    r = client.post("/internal/result/unit01", json=payload,
                     headers={"X-Callback-Token": "wrong"})
    assert r.status_code == 403
    r = client.post("/internal/result/unit01", json=payload,
                    headers={"X-Callback-Token": "unit-test-callback"})
    assert r.status_code == 200
    st = client.get("/api/firedrake/unit01", headers=AUTH).json()
    assert st["status"] == "done"
    assert st["runtime_ms"] == 1234.0
    # no params recorded (run was not called) -> no relative_l2
    assert st["relative_l2"] is None


def test_health_and_metrics(client):
    assert client.get("/api/health").json()["status"] == "ok"
    m = client.get("/api/metrics", headers=AUTH).json()
    assert "ndofs" in m
