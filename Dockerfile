# Multi-stage build
FROM python:3.11-slim AS builder
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends gcc g++ && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir uv
COPY requirements.txt .
RUN uv pip install --system -r requirements.txt

FROM python:3.11-slim
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends curl && rm -rf /var/lib/apt/lists/*
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

COPY . .

RUN mkdir -p data/raw models && \
    useradd -u 1000 -m appuser && \
    chown -R appuser:appuser /app
USER 1000

EXPOSE 8000
CMD ["uvicorn", "edge_api.server:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4", "--loop", "uvloop"]