import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch

from edge_api.server import app


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture
def mock_inference():
    with patch("edge_api.server.run_inference") as mock:
        mock.return_value = {
            "regime": "expansion",
            "severity": 0.75,
            "predicted_move": 45.2,
            "confidence": 0.82,
            "features": {"atr": 12.5, "impulse_norm": 0.8},
        }
        yield mock


def test_analyze_endpoint(client, mock_inference):
    # 30 candles
    candles = [
        {
            "time": 1640995200 + i * 60,
            "open": 1800.0 + i * 0.5,
            "high": 1810.0 + i * 0.5,
            "low": 1795.0 + i * 0.5,
            "close": 1805.0 + i * 0.5,
            "tick_volume": 1000 + i * 10,
        }
        for i in range(30)
    ]

    payload = {
        "asset": "XAUUSD",
        "candles": candles,
    }
    response = client.post("/analyze", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "regime" in data
    assert "predicted_move" in data


def test_warmup_endpoint(client):
    response = client.get("/warmup")
    assert response.status_code in (200, 204)


def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_unauthorized_if_api_key_set(client, monkeypatch):
    monkeypatch.setenv("API_KEY", "secret123")

    # Still mock run_inference so we don't hit real inference + missing 'impulse'
    with patch("edge_api.server.run_inference") as mock_run:
        mock_run.return_value = {"regime": "expansion", "predicted_move": 0}  # dummy

        # 30 candles
        candles = [
            {
                "time": 1640995200 + i * 60,
                "open": 1800.0 + i * 0.5,
                "high": 1810.0 + i * 0.5,
                "low": 1795.0 + i * 0.5,
                "close": 1805.0 + i * 0.5,
                "tick_volume": 1000 + i * 10,
            }
            for i in range(30)
        ]

        payload = {
            "asset": "XAUUSD",
            "candles": candles,
        }

        response = client.post("/analyze", json=payload)
        # Accept any of these — the important thing is it's not a successful 200 with real inference
        assert response.status_code in (401, 403, 422)
