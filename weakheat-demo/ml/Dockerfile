FROM python:3.11-slim

RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir numpy scipy matplotlib pytest \
    fastapi uvicorn pydantic kubernetes httpx

ENV PYTHONUNBUFFERED=1 \
    HOME=/tmp \
    OMP_NUM_THREADS=4

WORKDIR /work

CMD ["bash"]

