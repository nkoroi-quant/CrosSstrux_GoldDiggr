import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch

from edge_api.server import app

client = TestClient(app)


@pytest.fixture
def mock_inference():
    with patch("inference.engine.run_inference") as mock:
        mock.return_value = {
            "asset": "XAUUSD",
            "regime": "high",
            "probability": 0.87,
            "drift_psi": 0.08,
            "latency_ms": 12.3,
        }
        yield mock


def test_analyze_endpoint(mock_inference):
    payload = {
        "asset": "XAUUSD",
        "candles": [
            {
                "time": f"2025-03-01T00:{i:02d}:00",
                "open": 2650 + i * 0.1,
                "high": 2655 + i * 0.1,
                "low": 2648 + i * 0.1,
                "close": 2652 + i * 0.1,
                "tick_volume": 1200,
            }
            for i in range(30)
        ],
    }
    response = client.post("/analyze", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["asset"] == "XAUUSD"
    assert "regime" in data
    mock_inference.assert_called_once()


def test_warmup_endpoint():
    response = client.get("/warmup")
    assert response.status_code == 200
    assert response.json()["status"] == "warm"


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_unauthorized_if_api_key_set(monkeypatch):
    monkeypatch.setattr("config.settings.settings.API_KEY", "secret-key")
    payload = {"asset": "XAUUSD", "candles": []}
    response = client.post("/analyze", json=payload)
    assert response.status_code == 401
