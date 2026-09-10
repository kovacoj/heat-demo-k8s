"""Neural inference: one forward pass over all 26 frames for given params."""
import json
import os
import time

import numpy as np
import torch

from ml.model import CoordinateMLP  # noqa: F401  (for torch.load safety)
from ml.dataset import evaluate_fields

CHECKPOINT = os.environ.get("WEAKHEAT_CKPT", "/app/checkpoints/weakheat_best.pt")
OPERATORS_DIR = os.environ.get("WEAKHEAT_OPERATORS", "/app/data/operators")


class InferenceEngine:
    def __init__(self):
        ckpt = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
        self.model = CoordinateMLP(ckpt["in_dim"], ckpt["hidden"], ckpt["depth"])
        self.model.load_state_dict(ckpt["state_dict"])
        self.model.eval()
        self.n_params = self.model.num_parameters()

        coords = np.load(os.path.join(OPERATORS_DIR, "coords.npy"))
        self.coords = torch.tensor(coords, dtype=torch.float32)
        self.order = np.load(os.path.join(OPERATORS_DIR, "order.npy"))

        meta_path = os.environ.get("WEAKHEAT_META", "/app/checkpoints/model_meta.json")
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                self.meta = json.load(f)
        else:
            self.meta = {}

        metrics_path = os.environ.get(
            "WEAKHEAT_METRICS", "/app/presentation/metrics.json")
        self.metrics = {}
        if os.path.exists(metrics_path):
            with open(metrics_path) as f:
                self.metrics = json.load(f)

        self.times = torch.tensor(
            np.round(np.arange(0.0, 0.25 + 1e-12, 0.01), 10), dtype=torch.float32)

    def predict(self, x0, y0, sigma, alpha) -> dict:
        params = torch.tensor([[x0, y0, sigma, alpha]], dtype=torch.float32)
        t0 = time.perf_counter()
        u = evaluate_fields(self.model, self.coords, params, self.times)
        inference_ms = (time.perf_counter() - t0) * 1000.0
        # canonical 33x33 row-major (y, x) order
        frames = u[0][:, self.order].numpy().tolist()
        return {
            "times": self.times.numpy().tolist(),
            "frames": frames,
            "inference_ms": inference_ms,
            "shape": [33, 33],
        }
