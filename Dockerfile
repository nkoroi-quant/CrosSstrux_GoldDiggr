FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY core/ ./core/
COPY adapter/ ./adapter/
COPY inference/ ./inference/
COPY edge_api/ ./edge_api/
COPY data_layer/ ./data_layer/
COPY training/ ./training/
COPY config/ ./config/
COPY models/ ./models/
COPY run_training_pipeline.py .
COPY tools/ ./tools/

RUN mkdir -p data/raw models

EXPOSE 8000
CMD ["uvicorn", "edge_api.server:app", "--host", "0.0.0.0", "--port", "8000"]
