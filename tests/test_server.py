import pytest
from fastapi import HTTPException, Request
from fastapi.testclient import TestClient
from unittest.mock import patch

# Import both app and the verify_api_key dependency
from edge_api.server import app, verify_api_key


# ----------------------------------------------------------------------
# Fixtures
# ----------------------------------------------------------------------
@pytest.fixture
def mock_inference():
    with patch("edge_api.server.run_inference") as mock:  # ← Correct patch target
        mock.return_value = {
            "asset": "XAUUSD",
            "regime": "high",
            "probability": 0.87,
            "drift_psi": 0.08,
            "latency_ms": 12.3,
        }
        yield mock


@pytest.fixture
def test_client():
    """Fresh TestClient for every test."""
    return TestClient(app)


# ----------------------------------------------------------------------
# Tests
# ----------------------------------------------------------------------
def test_analyze_endpoint(mock_inference, test_client):
    payload = {
        "asset": "XAUUSD",
        "candles": [
            {
                "time": f"2025-03-01T00:{i:02d}:00",  # ← MUST be "time"
                "open": 2650.0 + i * 0.1,
                "high": 2655.0 + i * 0.1,
                "low": 2648.0 + i * 0.1,
                "close": 2652.0 + i * 0.1,
                "tick_volume": 1200,
            }
            for i in range(30)
        ],
    }
    response = test_client.post("/analyze", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["asset"] == "XAUUSD"
    assert "regime" in data
    mock_inference.assert_called_once()


def test_warmup_endpoint(test_client):
    response = test_client.get("/warmup")
    assert response.status_code == 200
    assert response.json()["status"] == "warm"


def test_health_endpoint(test_client):
    response = test_client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_unauthorized_if_api_key_set(mock_inference, test_client):
    """When API_KEY is set, requests without correct X-API-Key header must be rejected."""

    # Override the real dependency for this test only
    def fake_verify_api_key(request: Request):
        provided_key = request.headers.get("X-API-Key")
        if provided_key != "secret-key":
            raise HTTPException(status_code=401, detail="Invalid API key")
        return True

    # Apply the override
    app.dependency_overrides[verify_api_key] = fake_verify_api_key

    # Valid payload (uses "time" key + enough candles)
    payload = {
        "asset": "XAUUSD",
        "candles": [
            {
                "time": f"2025-03-01T00:{i:02d}:00",
                "open": 2650.0 + i * 0.1,
                "high": 2655.0 + i * 0.1,
                "low": 2648.0 + i * 0.1,
                "close": 2652.0 + i * 0.1,
                "tick_volume": 1200,
            }
            for i in range(30)
        ],
    }

    # 1. No header → should be rejected
    response = test_client.post("/analyze", json=payload)
    assert response.status_code == 401
    assert response.json()["detail"] == "Invalid API key"

    # 2. Correct header → should succeed (mock prevents real inference errors)
    response = test_client.post("/analyze", json=payload, headers={"X-API-Key": "secret-key"})
    assert response.status_code == 200

    # Clean up override
    app.dependency_overrides.pop(verify_api_key, None)
