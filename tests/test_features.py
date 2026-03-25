"""
Tests for core.features.feature_pipeline module.

These tests verify that the feature pipeline produces identical results
in both training and inference contexts.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timezone

from core.features.feature_pipeline import (
    compute_atr,
    validate_input_columns,
    build_features,
    get_feature_columns,
)


class TestComputeATR:
    """Tests for ATR computation."""
    
    def test_atr_basic(self):
        """Test basic ATR computation."""
        df = pd.DataFrame({
            "high": [1.10, 1.15, 1.20, 1.18, 1.25],
            "low": [1.00, 1.05, 1.10, 1.08, 1.15],
            "close": [1.05, 1.12, 1.18, 1.15, 1.22],
        })
        
        atr = compute_atr(df, period=3)
        
        assert len(atr) == len(df)
        assert atr.iloc[0] > 0  # First value should be computed with min_periods=1
        assert not atr.isna().any()
    
    def test_atr_reuse_existing(self):
        """Test that existing ATR column is reused."""
        df = pd.DataFrame({
            "high": [1.10, 1.15],
            "low": [1.00, 1.05],
            "close": [1.05, 1.12],
            "atr": [0.05, 0.06],  # Pre-existing ATR
        })
        
        atr = compute_atr(df)
        
        # Should return the existing column
        assert atr.equals(df["atr"])
    
    def test_atr_missing_columns(self):
        """Test error when required columns are missing."""
        df = pd.DataFrame({
            "high": [1.10, 1.15],
            "close": [1.05, 1.12],
            # Missing "low"
        })
        
        with pytest.raises(ValueError, match="Required column 'low' not found"):
            compute_atr(df)


class TestValidateInputColumns:
    """Tests for input validation and renaming."""
    
    def test_timestamp_rename(self):
        """Test that timestamp is renamed to time."""
        df = pd.DataFrame({
            "timestamp": pd.to_datetime(["2021-01-01", "2021-01-02"]),
            "open": [1.0, 1.1],
            "high": [1.1, 1.2],
            "low": [0.9, 1.0],
            "close": [1.05, 1.15],
        })
        
        result = validate_input_columns(df)
        
        assert "time" in result.columns
        assert "timestamp" not in result.columns
    
    def test_volume_rename(self):
        """Test that volume is renamed to tick_volume."""
        df = pd.DataFrame({
            "time": pd.to_datetime(["2021-01-01", "2021-01-02"]),
            "open": [1.0, 1.1],
            "high": [1.1, 1.2],
            "low": [0.9, 1.0],
            "close": [1.05, 1.15],
            "volume": [100, 200],
        })
        
        result = validate_input_columns(df)
        
        assert "tick_volume" in result.columns
        assert "volume" not in result.columns
    
    def test_missing_required_columns(self):
        """Test error when required columns are missing."""
        df = pd.DataFrame({
            "time": pd.to_datetime(["2021-01-01"]),
            "open": [1.0],
            # Missing high, low, close
        })
        
        with pytest.raises(ValueError, match="Missing required columns"):
            validate_input_columns(df)
    
    def test_time_conversion(self):
        """Test that time column is converted to datetime."""
        df = pd.DataFrame({
            "time": ["2021-01-01 00:00:00", "2021-01-01 00:01:00"],
            "open": [1.0, 1.1],
            "high": [1.1, 1.2],
            "low": [0.9, 1.0],
            "close": [1.05, 1.15],
        })
        
        result = validate_input_columns(df)
        
        assert pd.api.types.is_datetime64_any_dtype(result["time"])


class TestBuildFeatures:
    """Tests for feature building."""
    
    def create_sample_df(self, n=50):
        """Create a sample DataFrame for testing."""
        np.random.seed(42)  # For reproducibility
        
        return pd.DataFrame({
            "time": pd.date_range("2021-01-01", periods=n, freq="min", tz="UTC"),
            "open": np.random.randn(n).cumsum() + 100,
            "high": np.random.randn(n).cumsum() + 101,
            "low": np.random.randn(n).cumsum() + 99,
            "close": np.random.randn(n).cumsum() + 100,
            "tick_volume": np.random.randint(100, 1000, n),
            "impulse": np.random.randint(0, 2, n),
            "balance": np.random.randint(0, 2, n),
            "volatility_expansion": np.random.randn(n) * 0.01,
            "regime_stability": np.random.rand(n),
        })
    
    def test_build_features_basic(self):
        """Test basic feature building."""
        df = self.create_sample_df()
        
        result = build_features(df)
        
        # Check required columns
        expected_cols = ["time", "impulse_norm", "balance", "volatility_expansion", "regime_stability"]
        for col in expected_cols:
            assert col in result.columns, f"Missing column: {col}"
    
    def test_build_features_deterministic(self):
        """Test that feature building is deterministic."""
        df = self.create_sample_df()
        
        result1 = build_features(df)
        result2 = build_features(df)
        
        # Results should be identical
        pd.testing.assert_frame_equal(result1, result2)
    
    def test_build_features_with_timestamp(self):
        """Test feature building with timestamp column."""
        df = self.create_sample_df()
        df = df.rename(columns={"time": "timestamp"})
        
        result = build_features(df)
        
        assert "time" in result.columns
        assert "timestamp" not in result.columns
    
    def test_build_features_missing_computed(self):
        """Test error when computed columns are missing."""
        df = pd.DataFrame({
            "time": pd.date_range("2021-01-01", periods=10, freq="min"),
            "open": [1.0] * 10,
            "high": [1.1] * 10,
            "low": [0.9] * 10,
            "close": [1.05] * 10,
            # Missing impulse, balance, etc.
        })
        
        with pytest.raises(ValueError, match="Missing computed columns"):
            build_features(df)
    
    def test_impulse_norm_computation(self):
        """Test that impulse_norm is computed correctly."""
        df = pd.DataFrame({
            "time": pd.date_range("2021-01-01", periods=20, freq="min"),
            "open": [100.0] * 20,
            "high": [101.0] * 20,
            "low": [99.0] * 20,
            "close": [100.5] * 20,
            "impulse": [1] * 20,
            "balance": [0] * 20,
            "volatility_expansion": [0.01] * 20,
            "regime_stability": [0.5] * 20,
        })
        
        result = build_features(df)
        
        # impulse_norm = impulse / (atr + 1e-8)
        # With constant prices, ATR will be small
        assert "impulse_norm" in result.columns
        assert (result["impulse_norm"] > 0).all()


class TestGetFeatureColumns:
    """Tests for get_feature_columns function."""
    
    def test_feature_columns_list(self):
        """Test that feature columns list is correct."""
        cols = get_feature_columns()
        
        expected = ["impulse_norm", "balance", "volatility_expansion", "regime_stability"]
        assert cols == expected
    
    def test_feature_columns_consistency(self):
        """Test that feature columns are consistent."""
        cols1 = get_feature_columns()
        cols2 = get_feature_columns()
        
        assert cols1 == cols2


class TestTrainingInferenceParity:
    """
    Critical tests ensuring feature pipeline produces identical results
    in training and inference contexts.
    """
    
    def create_structured_df(self, n=100):
        """Create a DataFrame that has gone through structure pipeline."""
        np.random.seed(42)
        
        from core.structure.impulse import detect_impulse
        from core.structure.balance import detect_balance
        from core.structure.volatility import volatility_expansion
        from core.regimes.regime_classifier import classify_regime
        from core.regimes.stability import regime_stability
        
        df = pd.DataFrame({
            "time": pd.date_range("2021-01-01", periods=n, freq="min", tz="UTC"),
            "open": np.random.randn(n).cumsum() * 0.1 + 100,
            "high": np.random.randn(n).cumsum() * 0.1 + 100.5,
            "low": np.random.randn(n).cumsum() * 0.1 + 99.5,
            "close": np.random.randn(n).cumsum() * 0.1 + 100,
            "tick_volume": np.random.randint(100, 1000, n),
        })
        
        # Run structure pipeline
        df = detect_impulse(df)
        df = detect_balance(df)
        df = volatility_expansion(df)
        df = classify_regime(df)
        df = regime_stability(df)
        
        return df
    
    def test_training_inference_identical(self):
        """
        Test that the same input produces identical features in both contexts.
        This is the most critical test for the feature pipeline.
        """
        df = self.create_structured_df(n=100)
        
        # Simulate training context
        features_training = build_features(df)
        
        # Simulate inference context (same data)
        features_inference = build_features(df)
        
        # Must be identical
        pd.testing.assert_frame_equal(features_training, features_inference)
    
    def test_partial_data_consistency(self):
        """Test that partial data produces consistent features."""
        df_full = self.create_structured_df(n=100)
        df_partial = df_full.tail(50).reset_index(drop=True)
        
        features_full = build_features(df_full)
        features_partial = build_features(df_partial)
        
        # The overlapping rows should have identical features
        # (except for the first few rows that may have NaN due to rolling windows)
        overlap = min(len(features_full), len(features_partial))
        
        # Compare feature columns only (not time)
        feature_cols = ["impulse_norm", "balance", "volatility_expansion", "regime_stability"]
        
        for col in feature_cols:
            # Compare last 'overlap' rows of full with all of partial
            full_values = features_full[col].tail(overlap).values
            partial_values = features_partial[col].tail(overlap).values
            
            # Use allclose for floating point comparison
            np.testing.assert_allclose(full_values, partial_values, rtol=1e-10)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
