# Derived from the working heat-demo Firedrake image -- no Firedrake rebuild.
ARG BASE_IMAGE=cerit.io/kovacoj1/heat-firedrake:20260909-081208
FROM ${BASE_IMAGE}

COPY firedrake/worker.py firedrake/solve_case.py /app/
COPY data/operators/order.npy /opt/weakheat/order.npy

ENV PYTHONUNBUFFERED=1 \
    HOME=/tmp

WORKDIR /app

# mesh resolution, dt, t_end and solver settings are fixed in solve_case.py;
# the container entrypoint only accepts the four validated physical
# parameters via the API-created Kubernetes Job.
ENTRYPOINT ["python3", "/app/worker.py"]
