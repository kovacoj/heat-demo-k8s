FROM firedrakeproject/firedrake-vanilla-default:latest


WORKDIR /home/firedrake/app


# Install web-server packages somewhere independent
# of whichever UID Kubernetes runs the container as.
RUN python3 -m pip install \
    --target=/opt/python-packages \
    fastapi \
    uvicorn


ENV PYTHONPATH="/opt/python-packages:${PYTHONPATH}"

ENV PYTHONUNBUFFERED=1

ENV HOME=/tmp

ENV OMP_NUM_THREADS=1

ENV OPENBLAS_NUM_THREADS=1

ENV MKL_NUM_THREADS=1


COPY app.py .
COPY worker.py .


EXPOSE 8080


CMD ["python3", "-m", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]
