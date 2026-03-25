"""
Tests for edge_api.server module.
"""

import os
import sys
import json
import tempfile
import shutil
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timezone
from fastapi.testclient import TestClient

# Add parent to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from edge_api.server import (
    app,
    compute_psi,
    select_regime,
    classify_severity,
    get_model_with_fallback,
    FALLBACK_CHAIN,
    GOVERNANCE_THRESHOLDS,
    RISK_MULTIPLIER,
    MIN_DRIFT_SAMPLES,
)

client = TestClient(app)


class TestPSIComputation:
    """Tests for PSI computation."""
    
    def test_psi_identical_distributions(self):
        """PSI should be ~0 for identical distributions."""
        expected = [1.0, 2.0, 3.0, 4.0, 5.0] * 20
        actual = [1.0, 2.0, 3.0, 4.0, 5.0] * 20
        
        psi = compute_psi(expected, actual)
        
        assert psi >= 0
        assert psi < 0.1  # Should be very small
    
    def test_psi_different_distributions(self):
        """PSI should be higher for different distributions."""
        expected = np.random.normal(0, 1, 1000).tolist()
        actual = np.random.normal(2, 1, 1000).tolist()
        
        psi = compute_psi(expected, actual)
        
        assert psi > 0.1  # Should be noticeably higher
    
    def test_psi_insufficient_samples(self):
        """PSI should be 0 for insufficient samples."""
        expected = [1.0, 2.0, 3.0]
        actual = [1.0]  # Only 1 sample
        
        psi = compute_psi(expected, actual)
        
        assert psi == 0.0


class TestSelectRegime:
    """Tests for regime selection."""
    
    def test_select_low_regime(self):
        """Test selecting low regime."""
        # Mock metadata
        from edge_api.server import asset_metadata
        asset_metadata["TEST"] = {"low_threshold": 0.4, "high_threshold": 0.8}
        
        regime = select_regime("TEST", 0.2)
        assert regime == "low"
    
    def test_select_mid_regime(self):
        """Test selecting mid regime."""
        from edge_api.server import asset_metadata
        asset_metadata["TEST"] = {"low_threshold": 0.4, "high_threshold": 0.8}
        
        regime = select_regime("TEST", 0.6)
        assert regime == "mid"
    
    def test_select_high_regime(self):
        """Test selecting high regime."""
        from edge_api.server import asset_metadata
        asset_metadata["TEST"] = {"low_threshold": 0.4, "high_threshold": 0.8}
        
        regime = select_regime("TEST", 0.9)
        assert regime == "high"


class TestClassifySeverity:
    """Tests for severity classification."""
    
    def test_severity_levels(self):
        """Test all severity levels."""
        # low regime thresholds: [0.10, 0.25, 0.50]
        
        assert classify_severity("low", 0.05) == 0  # Below t1
        assert classify_severity("low", 0.15) == 1  # Between t1 and t2
        assert classify_severity("low", 0.35) == 2  # Between t2 and t3
        assert classify_severity("low", 0.60) == 3  # Above t3


class TestFallbackChain:
    """Tests for model fallback logic."""
    
    def test_fallback_chain_defined(self):
        """Test that fallback chain is defined for all regimes."""
        assert "low" in FALLBACK_CHAIN
        assert "mid" in FALLBACK_CHAIN
        assert "high" in FALLBACK_CHAIN


class TestHealthEndpoint:
    """Tests for health check endpoint."""
    
    def test_health_check(self):
        """Test health check returns expected fields."""
        response = client.get("/health")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "status" in data
        assert data["status"] == "healthy"
        assert "timestamp" in data
        assert "loaded_assets" in data


class TestMetricsEndpoint:
    """Tests for metrics endpoint."""
    
    def test_metrics(self):
        """Test metrics endpoint returns expected structure."""
        response = client.get("/metrics")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "assets_loaded" in data
        assert "assets" in data


class TestPredictEndpoint:
    """Tests for predict endpoint."""
    
    def create_sample_candles(self, n=50):
        """Create sample candle data."""
        np.random.seed(42)
        
        candles = []
        base_time = datetime(2021, 1, 1, 12, 0, 0)
        
        for i in range(n):
            close = 100 + np.random.randn() * 0.5
            candles.append({
                "timestamp": (base_time.replace(minute=i)).isoformat(),
                "open": close - 0.1,
                "high": close + 0.2,
                "low": close - 0.2,
                "close": close,
                "volume": np.random.randint(100, 1000),
            })
        
        return candles
    
    def test_predict_no_candles(self):
        """Test predict with no candles returns error."""
        response = client.post("/predict", json={
            "asset": "TEST",
            "candles": []
        })
        
        assert response.status_code == 400
        assert "No candles" in response.json()["detail"]
    
    def test_predict_insufficient_candles(self):
        """Test predict with insufficient candles returns error."""
        response = client.post("/predict", json={
            "asset": "TEST",
            "candles": self.create_sample_candles(n=5)
        })
        
        assert response.status_code == 400
        assert "Insufficient candles" in response.json()["detail"]
    
    def test_predict_asset_not_found(self):
        """Test predict for non-existent asset returns error."""
        response = client.post("/predict", json={
            "asset": "NONEXISTENT",
            "candles": self.create_sample_candles(n=50)
        })
        
        assert response.status_code == 400
        assert "No models found" in response.json()["detail"]


class TestResponseStructure:
    """Tests for response structure validation."""
    
    def test_response_model_fields(self):
        """Test that response model has all required fields."""
        from edge_api.server import PredictResponse
        
        # Create a sample response
        response_data = {
            "asset": "TEST",
            "regime": "low",
            "probability": 0.75,
            "transition_probability": 0.3,
            "cdi": 0.15,
            "severity": 1,
            "confirmed_elevated": False,
            "consecutive_elevated": 0,
            "risk_multiplier": 1.0,
            "top_drift_feature": "impulse_norm",
            "rolling_samples": 100,
        }
        
        response = PredictResponse(**response_data)
        
        assert response.asset == "TEST"
        assert response.regime == "low"
        assert 0 <= response.probability <= 1
        assert 0 <= response.cdi
        assert response.severity in [0, 1, 2, 3]


class TestRiskMultiplier:
    """Tests for risk multiplier logic."""
    
    def test_risk_multiplier_values(self):
        """Test that risk multiplier values are correct."""
        assert RISK_MULTIPLIER[0] == 1.00
        assert RISK_MULTIPLIER[1] == 0.75
        assert RISK_MULTIPLIER[2] == 0.50
        assert RISK_MULTIPLIER[3] == 0.00


class TestGovernanceThresholds:
    """Tests for governance thresholds."""
    
    def test_thresholds_defined_for_all_regimes(self):
        """Test that thresholds are defined for all regimes."""
        assert "low" in GOVERNANCE_THRESHOLDS
        assert "mid" in GOVERNANCE_THRESHOLDS
        assert "high" in GOVERNANCE_THRESHOLDS
        
        # Each should have 3 thresholds
        for regime in ["low", "mid", "high"]:
            assert len(GOVERNANCE_THRESHOLDS[regime]) == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
