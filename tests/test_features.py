import pytest
import pandas as pd
import numpy as np

from core.features.feature_pipeline import (
    compute_atr,
    validate_input_columns,
    build_features,
    get_extended_feature_columns,
)


@pytest.fixture
def sample_ohlc():
    return pd.DataFrame(
        {
            "time": pd.date_range("2025-01-01", periods=100, freq="min"),
            "open": 2650.0 + np.random.randn(100) * 0.5,
            "high": 2655.0 + np.random.randn(100) * 0.5,
            "low": 2648.0 + np.random.randn(100) * 0.5,
            "close": 2652.0 + np.random.randn(100) * 0.5,
            "tick_volume": 1000 + np.random.randint(0, 500, 100),
        }
    )


def test_atr_basic(sample_ohlc):
    result = compute_atr(sample_ohlc)
    assert "atr" in result.columns
    assert result["atr"].iloc[0] >= 0


def test_atr_reuse_existing(sample_ohlc):
    sample_ohlc["atr"] = 0.05
    result = compute_atr(sample_ohlc)
    assert result["atr"].equals(sample_ohlc["atr"])


def test_atr_missing_columns(sample_ohlc):
    df = sample_ohlc.drop(columns=["low"])
    with pytest.raises(KeyError):
        compute_atr(df)


def test_validate_input_columns(sample_ohlc):
    result = validate_input_columns(sample_ohlc)
    assert "time" in result.columns
    assert "tick_volume" in result.columns


def test_build_features_basic(sample_ohlc):
    result = build_features(sample_ohlc)
    assert "impulse_norm" in result.columns
    assert len(result) == len(sample_ohlc)


def test_build_features_deterministic(sample_ohlc):
    r1 = build_features(sample_ohlc.copy())
    r2 = build_features(sample_ohlc.copy())
    pd.testing.assert_frame_equal(r1, r2)


def test_impulse_norm_computation(sample_ohlc):
    result = build_features(sample_ohlc)
    assert (result["impulse_norm"] >= 0).all()


def test_feature_columns_list():
    cols = get_extended_feature_columns()
    assert "impulse_norm" in cols


def test_training_inference_parity(sample_ohlc):
    full = build_features(sample_ohlc)
    partial = build_features(sample_ohlc.iloc[:50])
    # relaxed check for parity on overlapping part
    pd.testing.assert_frame_equal(full.iloc[:50], partial, check_dtype=False)
