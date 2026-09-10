# Small ordinary Python image -- no Firedrake needed here.
FROM python:3.11-slim

RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir fastapi uvicorn kubernetes pydantic numpy scipy

ENV PYTHONUNBUFFERED=1 \
    HOME=/tmp \
    PYTHONPATH=/app \
    OMP_NUM_THREADS=2

WORKDIR /app

COPY api/ /app/api/
COPY ml/model.py ml/dataset.py /app/ml/
COPY data/operators/ /app/data/operators/
COPY checkpoints/ /app/checkpoints/
COPY presentation/metrics.json /app/presentation/metrics.json

EXPOSE 8000

CMD ["python3", "-m", "uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
