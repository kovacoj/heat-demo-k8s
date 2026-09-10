"""Pydantic schemas with strict parameter clamping (spec section 19).

The public API can only vary the four physical parameters, and only inside
the training domain.  No image, mesh size, CPU or command is ever
client-selectable.
"""
from pydantic import BaseModel, field_validator

PARAM_LIMITS = {
    "x0": (0.25, 0.75),
    "y0": (0.25, 0.75),
    "sigma": (0.04, 0.10),
    "alpha": (0.005, 0.02),
}


class HeatParams(BaseModel):
    x0: float
    y0: float
    sigma: float
    alpha: float

    @field_validator("x0")
    @classmethod
    def _v_x0(cls, v):
        lo, hi = PARAM_LIMITS["x0"]
        if not (lo <= v <= hi):
            raise ValueError(f"x0 must be in [{lo}, {hi}]")
        return v

    @field_validator("y0")
    @classmethod
    def _v_y0(cls, v):
        lo, hi = PARAM_LIMITS["y0"]
        if not (lo <= v <= hi):
            raise ValueError(f"y0 must be in [{lo}, {hi}]")
        return v

    @field_validator("sigma")
    @classmethod
    def _v_sigma(cls, v):
        lo, hi = PARAM_LIMITS["sigma"]
        if not (lo <= v <= hi):
            raise ValueError(f"sigma must be in [{lo}, {hi}]")
        return v

    @field_validator("alpha")
    @classmethod
    def _v_alpha(cls, v):
        lo, hi = PARAM_LIMITS["alpha"]
        if not (lo <= v <= hi):
            raise ValueError(f"alpha must be in [{lo}, {hi}]")
        return v


class NNResponse(BaseModel):
    times: list[float]
    frames: list[list[float]]
    inference_ms: float
    shape: list[int]


class RunResponse(BaseModel):
    job_id: str
    status: str


class JobStatus(BaseModel):
    job_id: str
    status: str  # pending | running | done | error
    times: list[float] | None = None
    frames: list[list[float]] | None = None
    runtime_ms: float | None = None
    shape: list[int] | None = None
    relative_l2: float | None = None
    error: str | None = None
