FROM opencfd/openfoam-default


# The supervisor script needs python3 (the OpenFOAM image
# ships without it).
RUN apt-get update \
    && apt-get install -y --no-install-recommends python3 \
    && rm -rf /var/lib/apt/lists/*


COPY worker_of.py /opt/worker_of.py


# OpenFOAM needs a writable HOME (MPI session directories).
ENV HOME=/tmp
