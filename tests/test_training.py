import pytest
import pandas as pd
import tempfile
import os
from unittest.mock import patch

from training.train import train_asset, create_continuation_target, assign_regime


@pytest.fixture
def sample_df():
    return pd.DataFrame(
        {
            "time": pd.date_range("2025-01-01", periods=100, freq="min"),
            "open": 2650.0 + pd.Series(range(100)) * 0.1,
            "high": 2655.0 + pd.Series(range(100)) * 0.1,
            "low": 2648.0 + pd.Series(range(100)) * 0.1,
            "close": 2652.0 + pd.Series(range(100)) * 0.1,
            "tick_volume": 1000 + pd.Series(range(100)),
            "impulse_norm": pd.Series(range(100)) / 100.0,
        }
    )


def test_create_continuation_target(sample_df):
    result = create_continuation_target(sample_df)
    assert "continuation_y" in result.columns
    assert (
        result["continuation_y"].dtype == "int64" or str(result["continuation_y"].dtype) == "int32"
    )


def test_assign_regime(sample_df):
    impulse_col = "impulse_norm"
    low, high = 0.3, 0.7
    result = assign_regime(sample_df, impulse_col, low, high)
    assert "model_regime" in result.columns


@pytest.mark.parametrize("force_retrain", [True, False])
def test_train_asset_smoke(sample_df, force_retrain, tmp_path, monkeypatch):
    monkeypatch.setattr("training.train.DATA_DIR", str(tmp_path))
    monkeypatch.setattr("training.train.MODEL_DIR", str(tmp_path / "models"))

    with patch("pandas.read_parquet", return_value=sample_df):
        with patch("os.path.exists", return_value=False):
            success = train_asset(
                asset="XAUUSD",
                max_rows=50,
                force=True,
                force_retrain=force_retrain,
            )
            assert success is True


def test_deprecated_wrapper_warning():
    import warnings

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        import run_training_pipeline

        assert any("deprecated" in str(m.message).lower() for m in w)
