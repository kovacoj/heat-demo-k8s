# weakheat-demo

End-to-end demonstration comparing a **Firedrake transient FEM solve** with a
**weak-form neural surrogate** for the 2D heat equation.

```
                          GitHub Pages
                               |
                               | HTTPS
                               v
                         FastAPI / CERIT
                        /              \
                       /                \
                      v                  v
             Kubernetes Job         PyTorch
                Firedrake            model
                    |                  |
                    +--------+---------+
                             |
                             v
                       common FE field
                       33 x 33 x time
```

## The mathematical idea

The strong-form PINN residual requires the Laplacian of the network:

```
u_t - alpha (u_xx + u_yy)
```

Integration by parts removes the second derivatives. Testing the heat
equation against the complete Firedrake CG1 basis gives

```
Firedrake weak form

    int u_t v dx  +  alpha int grad(u) . grad(v) dx  = 0

                    |  FE basis
                    v

    M dc/dt + alpha K c = 0

                    |  neural coefficients
                    v

    M dc_theta/dt + alpha K c_theta  =  r_theta
```

Firedrake assembles `M` (mass) and `K` (stiffness) once. PyTorch supplies
the neural field `c_theta(t)` and its **first** time derivative via
autograd in `t` only. The residual `r_theta` never evaluates second spatial
derivatives of the network — the Laplacian has been moved onto the FE test
functions. This is the central methodological statement of the demo.

## Problem

```
du/dt - alpha laplace(u) = 0        in (0,1)^2
grad(u) . n = 0                      (natural Neumann BC)
u(x,y,0) = exp(-((x-x0)^2+(y-y0)^2) / (2 sigma^2))

x0, y0 in [0.25, 0.75], sigma in [0.04, 0.10], alpha in [0.005, 0.02]
```

Discretisation: 32x32 CG1 (1089 dofs), implicit Euler, dt = 0.005,
t_end = 0.25, 26 output frames.

## Layout

```
firedrake/   solve_case, export_operators, generate_dataset, worker (K8s Job)
ml/          model, dataset, weak_loss, train, evaluate, benchmark
api/         FastAPI controller (deployed on CERIT)
k8s/         rbac, deployment, service, ingress (namespace kovacovsky-ns)
frontend/weakheat/   static GitHub Pages page (../frontend in this repo)
scripts/     build/push/deploy helpers
```

## Workflow (all compute runs in Docker)

```
make dataset            # Firedrake: export M, K + 200-case ground truth
make train              # weak-form surrogate (data + IC + weak residual)
make train-data-only    # supervised-only baseline (same architecture)
make evaluate           # held-out metrics + presentation figures
make test               # unit tests
```

Training objective:

```
L = L_data + lambda_weak * L_weak + L_IC

L_data : nodal MSE against the Firedrake reference
L_weak : mean( (M du/dt + alpha K u)^2 / m )     (lumped-mass scaled)
L_IC   : exact Gaussian at t = 0
```

`lambda_weak` is chosen after a supervised warm-up so the weak term
contributes ~20% of the total loss (its magnitude is measured, not
guessed).

## Evaluation

Held-out metrics (20 unseen configurations), from
`presentation/metrics.json`:

```
median / p90 / max relative FE L2 error   e = sqrt(e^T M e / u^T M u)
mass-conservation drift                   m(t) = 1^T M u(t)
scaled weak residual norm
NN inference runtime (26 frames) vs Firedrake transient runtime
```

## Deployment (CERIT Kubernetes, namespace kovacovsky-ns)

All resources are prefixed `weakheat-`:

```
weakheat-api          Deployment + Service + Ingress (FastAPI + PyTorch CPU)
weakheat-api          ServiceAccount + Role (create Jobs)
weakheat-fd-*         Kubernetes Jobs running the Firedrake worker
```

The browser presses **Run** -> neural inference returns in milliseconds
while a real Kubernetes Job solves the same configuration with Firedrake;
both fields animate over the same time slider with the relative L2 error
displayed. Public endpoints carry no token (compute is bounded: fixed mesh,
fixed solver, max 2 concurrent Firedrake Jobs); only the internal Job
callback requires the `weakheat-secrets` callback token.

No analytical solution is used anywhere: Firedrake is the reference.
No metric in the presentation is fabricated.
