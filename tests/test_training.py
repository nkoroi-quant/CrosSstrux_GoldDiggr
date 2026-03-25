"""
Tests for training pipeline.
"""

import os
import json
import tempfile
import shutil
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timezone

from sklearn.pipeline import Pipeline

# Import training module functions
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from run_training_pipeline import (
    compute_regime_thresholds,
    assign_regime,
    create_continuation_target,
    create_transition_target,
    train_regime_model,
    train_transition_model,
    save_drift_baseline,
)
from core.features.feature_pipeline import get_feature_columns


class TestRegimeThresholds:
    """Tests for regime threshold computation."""
    
    def test_compute_thresholds(self):
        """Test threshold computation with known quantiles."""
        df = pd.DataFrame({
            "impulse_norm": list(range(100))  # 0-99
        })
        
        low_q, high_q = compute_regime_thresholds(df, "impulse_norm")
        
        # 40th percentile of 0-99 should be ~39.6
        assert 39 <= low_q <= 40
        # 80th percentile of 0-99 should be ~79.2
        assert 79 <= high_q <= 80
    
    def test_assign_regime(self):
        """Test regime assignment."""
        df = pd.DataFrame({
            "impulse_norm": [10, 50, 90]
        })
        
        df = assign_regime(df, "impulse_norm", low_q=40, high_q=80)
        
        assert df["model_regime"].iloc[0] == "low"
        assert df["model_regime"].iloc[1] == "mid"
        assert df["model_regime"].iloc[2] == "high"


class TestContinuationTarget:
    """Tests for continuation target creation."""
    
    def test_continuation_target(self):
        """Test continuation target logic."""
        df = pd.DataFrame({
            "impulse": [0, 1, 0, 1, 0]
        })
        
        df = create_continuation_target(df, "impulse_norm")
        
        # continuation_y = next impulse value (shifted)
        # Last row should be 0 (filled NaN)
        assert df["continuation_y"].iloc[-1] == 0
        
        # Other rows should be 0 or 1
        assert df["continuation_y"].iloc[0] in [0, 1]
        
        # Check values are shifted correctly
        assert df["continuation_y"].iloc[0] == df["impulse"].iloc[1]


class TestTransitionTarget:
    """Tests for transition target creation."""
    
    def test_transition_target(self):
        """Test transition target logic."""
        df = pd.DataFrame({
            "model_regime": ["low", "low", "low", "mid", "mid", "high"]
        })
        
        df = create_transition_target(df)
        
        # Check regime encoding
        assert "regime_encoded" in df.columns
        assert df["regime_encoded"].iloc[0] == 0  # low
        assert df["regime_encoded"].iloc[3] == 1  # mid
        assert df["regime_encoded"].iloc[5] == 2  # high
        
        # Check transition detection
        assert "transition_y" in df.columns
        # Row 2 should detect transition to mid
        assert df["transition_y"].iloc[2] == 1


class TestTrainRegimeModel:
    """Tests for regime model training."""
    
    def test_train_regime_model(self):
        """Test training a regime model."""
        np.random.seed(42)
        
        X = pd.DataFrame({
            "impulse_norm": np.random.randn(100),
            "balance": np.random.randint(0, 2, 100),
            "volatility_expansion": np.random.randn(100) * 0.01,
            "regime_stability": np.random.rand(100),
        })
        y = pd.Series(np.random.randint(0, 2, 100))
        
        model = train_regime_model(X, y)
        
        assert isinstance(model, Pipeline)
        
        # Test prediction
        proba = model.predict_proba(X.head(5))
        assert proba.shape == (5, 2)
        assert np.all((proba >= 0) & (proba <= 1))


class TestTrainTransitionModel:
    """Tests for transition model training."""
    
    def test_train_transition_model(self):
        """Test training a transition model."""
        np.random.seed(42)
        
        X = pd.DataFrame({
            "impulse_norm": np.random.randn(100),
            "balance": np.random.randint(0, 2, 100),
            "volatility_expansion": np.random.randn(100) * 0.01,
            "regime_stability": np.random.rand(100),
            "regime_encoded": np.random.randint(0, 3, 100),
        })
        y = pd.Series(np.random.randint(0, 2, 100))
        
        model = train_transition_model(X, y)
        
        assert isinstance(model, Pipeline)
        
        # Test prediction
        proba = model.predict_proba(X.head(5))
        assert proba.shape == (5, 2)


class TestSaveDriftBaseline:
    """Tests for drift baseline saving."""
    
    def test_save_drift_baseline(self, tmp_path):
        """Test saving drift baseline."""
        X = pd.DataFrame({
            "impulse_norm": [0.1, 0.2, 0.3],
            "balance": [0, 1, 0],
            "volatility_expansion": [0.01, 0.02, 0.03],
            "regime_stability": [0.5, 0.6, 0.7],
        })
        
        output_path = tmp_path / "drift_baseline_test.json"
        
        # Temporarily change FEATURE_COLUMNS
        import run_training_pipeline as rtp
        original_cols = rtp.FEATURE_COLUMNS
        rtp.FEATURE_COLUMNS = list(X.columns)
        
        try:
            save_drift_baseline(X, str(output_path))
            
            assert output_path.exists()
            
            with open(output_path) as f:
                baseline = json.load(f)
            
            assert "impulse_norm" in baseline
            assert baseline["impulse_norm"] == [0.1, 0.2, 0.3]
        finally:
            rtp.FEATURE_COLUMNS = original_cols


class TestEndToEndTraining:
    """End-to-end tests for the training pipeline."""
    
    def create_sample_data(self, n=200):
        """Create sample data that can go through full pipeline."""
        np.random.seed(42)
        
        from core.structure.impulse import detect_impulse
        from core.structure.balance import detect_balance
        from core.structure.volatility import volatility_expansion
        from core.regimes.regime_classifier import classify_regime
        from core.regimes.stability import regime_stability
        from core.features.feature_pipeline import build_features
        
        df = pd.DataFrame({
            "time": pd.date_range("2021-01-01", periods=n, freq="min", tz="UTC"),
            "open": np.random.randn(n).cumsum() * 0.1 + 100,
            "high": np.random.randn(n).cumsum() * 0.1 + 100.5,
            "low": np.random.randn(n).cumsum() * 0.1 + 99.5,
            "close": np.random.randn(n).cumsum() * 0.1 + 100,
            "tick_volume": np.random.randint(100, 1000, n),
        })
        
        # Run full pipeline
        df = detect_impulse(df)
        df = detect_balance(df)
        df = volatility_expansion(df)
        df = classify_regime(df)
        df = regime_stability(df)
        df = build_features(df)
        
        return df
    
    def test_full_pipeline_integration(self, tmp_path):
        """Test full training pipeline with temporary directories."""
        import run_training_pipeline as rtp
        
        # Setup temp directories
        data_dir = tmp_path / "data" / "raw"
        model_dir = tmp_path / "models"
        data_dir.mkdir(parents=True)
        
        # Create and save sample data
        df = self.create_sample_data(n=300)
        data_path = data_dir / "TEST_M1.parquet"
        df.to_parquet(data_path, index=False)
        
        # Update module paths
        original_data_dir = rtp.DATA_DIR
        original_model_dir = rtp.MODEL_DIR
        rtp.DATA_DIR = str(data_dir)
        rtp.MODEL_DIR = str(model_dir)
        
        try:
            # Run training
            success = rtp.train_asset("TEST", max_rows=1000, force_retrain=True)
            
            assert success is True
            
            # Check outputs
            asset_path = model_dir / "TEST"
            assert asset_path.exists()
            
            # Check metadata
            metadata_path = asset_path / "metadata.json"
            assert metadata_path.exists()
            
            with open(metadata_path) as f:
                metadata = json.load(f)
            
            assert "low_threshold" in metadata
            assert "high_threshold" in metadata
            assert "feature_columns" in metadata
            assert "trained_regimes" in metadata
            
            # Check model files
            for regime in metadata["trained_regimes"]:
                model_path = asset_path / f"{regime}_model.pkl"
                assert model_path.exists()
                
                baseline_path = asset_path / f"drift_baseline_{regime}.json"
                assert baseline_path.exists()
            
            # Check transition model
            transition_path = asset_path / "transition_model.pkl"
            assert transition_path.exists()
            
        finally:
            rtp.DATA_DIR = original_data_dir
            rtp.MODEL_DIR = original_model_dir


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
