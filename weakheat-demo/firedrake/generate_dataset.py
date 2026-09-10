"""Generate the Firedrake ground-truth dataset for the neural surrogate.

Random parameter configurations are sampled uniformly inside the training
domain and each is solved with the full Firedrake transient solver.  The
Firedrake solution is the high-fidelity reference; no analytical solution
is used anywhere.

Output: data/raw/heat_dataset.npz with
    params            [Ncases, 4]        (x0, y0, sigma, alpha)
    times             [Ntimes]
    solutions         [Ncases, Ntimes, Ndofs]  (float32, native dof order)
    coords            [Ndofs, 2]
    solver_runtime_ms [Ncases]
    n_train / n_val / n_test
"""
import os
import time

import numpy as np

from solve_case import run_simulation, CaseParams, PARAM_LIMITS, TIMES

N_TRAIN, N_VAL, N_TEST = 160, 20, 20
SEED = 42
OUT_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "raw", "heat_dataset.npz")


def sample_params(rng, n):
    lo = np.array([PARAM_LIMITS[k][0] for k in ("x0", "y0", "sigma", "alpha")])
    hi = np.array([PARAM_LIMITS[k][1] for k in ("x0", "y0", "sigma", "alpha")])
    return lo + (hi - lo) * rng.random((n, 4))


def main():
    rng = np.random.default_rng(SEED)
    n_total = N_TRAIN + N_VAL + N_TEST
    all_params = sample_params(rng, n_total)

    # warm-up solve (JIT / assembly caching) so runtimes are representative
    _, _, _ = run_simulation(CaseParams(*all_params[0]))

    solutions = []
    runtimes = []
    t_start = time.perf_counter()
    for i, prm in enumerate(all_params):
        _, sols, rt_ms = run_simulation(CaseParams(*prm))
        solutions.append(sols.astype(np.float32))
        runtimes.append(rt_ms)
        if (i + 1) % 20 == 0 or i == n_total - 1:
            print(f"[{i+1}/{n_total}] last runtime {rt_ms:.0f} ms "
                  f"({time.perf_counter()-t_start:.0f} s elapsed)")

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    op_dir = os.path.join(os.path.dirname(__file__), "..", "data", "operators")
    coords = np.load(os.path.join(op_dir, "coords.npy"))
    np.savez_compressed(
        OUT_PATH,
        params=all_params.astype(np.float32),
        times=TIMES.astype(np.float32),
        solutions=np.stack(solutions),
        coords=coords,
        solver_runtime_ms=np.array(runtimes, dtype=np.float64),
        n_train=N_TRAIN, n_val=N_VAL, n_test=N_TEST,
    )
    print(f"saved {n_total} cases -> {os.path.abspath(OUT_PATH)}")

    # scientific sanity: peak decay + total heat conservation via M
    import scipy.sparse as sp
    op_dir = os.path.join(os.path.dirname(__file__), "..", "data", "operators")
    M = sp.load_npz(os.path.join(op_dir, "M.npz"))
    S = np.stack(solutions)          # [C, T, N]
    # total heat m(t) = 1^T M u(t); 1^T M is the lumped mass row
    mass = np.einsum("ctn,n->ct", S, np.asarray(M.sum(axis=1)).ravel())
    drift = np.abs(mass - mass[:, :1]) / np.maximum(mass[:, :1], 1e-12)
    print(f"max relative mass drift over all cases/times: {drift.max():.3e}")
    assert drift.max() < 1e-4, "Firedrake solver does not conserve total heat"
    print(f"median solver runtime: {np.median(runtimes):.1f} ms")


if __name__ == "__main__":
    main()
