.PHONY: help install test lint format collect train serve docker-build docker-run clean

help:
	@echo "CrossStrux v2 - Available Commands:"
	@echo ""
	@echo "  make install       - Install Python dependencies"
	@echo "  make test          - Run all tests"
	@echo "  make lint          - Run linting checks"
	@echo "  make format        - Format code with black"
	@echo "  make collect       - Run MT5 data collector"
	@echo "  make train         - Run training pipeline for XAUUSD"
	@echo "  make serve         - Start inference server"
	@echo "  make docker-build  - Build Docker image"
	@echo "  make docker-run    - Run Docker container"
	@echo ""

install:
	pip install -r requirements.txt

test:
	pytest tests/ -v --tb=short

lint:
	flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics
	flake8 . --count --exit-zero --max-complexity=10 --max-line-length=127 --statistics

format:
	black .

collect:
	python -m data_layer.collector --assets XAUUSD

train:
	python run_training_pipeline.py --assets XAUUSD

train-force:
	python run_training_pipeline.py --assets XAUUSD --force-retrain

serve:
	uvicorn edge_api.server:app --host 0.0.0.0 --port 8000 --reload

docker-build:
	docker build -t crossstrux-v2:latest .

docker-run:
	docker run -d --name crossstrux-v2 -p 8000:8000 \
		-v $(PWD)/models:/app/models \
		-v $(PWD)/config:/app/config \
		-v $(PWD)/data:/app/data \
		crossstrux-v2:latest

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type f -name "*.pyo" -delete 2>/dev/null || true
