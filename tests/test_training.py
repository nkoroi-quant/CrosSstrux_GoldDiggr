import pytest
import pandas as pd
from unittest.mock import patch

from training.train import create_continuation_target, train_asset
from core.regimes.regime_classifier import classify_regime
from core.features.feature_pipeline import build_features


@pytest.fixture
def sample_df():
    df = pd.DataFrame(
        {
            "time": pd.date_range("2024-01-01", periods=100, freq="min"),
            "open": range(100),
            "high": range(100, 200),
            "low": range(90, 190),
            "close": range(95, 195),
            "tick_volume": range(1000, 1100),
        }
    )
    return df  # time as column (required by validate_input_columns)


def test_create_continuation_target(sample_df):
    target_df = create_continuation_target(sample_df, horizon=2)
    assert len(target_df) == len(sample_df)
    assert "continuation_y" in target_df.columns


def test_assign_regime(sample_df):
    # build_features adds impulse_norm + atr
    df_with_features = build_features(sample_df.copy())
    # classify_regime needs impulse + balance columns
    df_with_features["impulse"] = 1
    df_with_features["balance"] = 0
    df_with_regime = classify_regime(df_with_features)
    assert "regime" in df_with_regime.columns


@pytest.mark.parametrize("force", [True, False])
def test_train_asset_smoke(sample_df, tmp_path, monkeypatch, force):
    data_dir = tmp_path / "data"
    model_dir = tmp_path / "models"
    data_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr("training.train.DATA_DIR", str(data_dir))
    monkeypatch.setattr("training.train.MODEL_DIR", str(model_dir))

    sample_df.to_parquet(data_dir / "XAUUSD_M1.parquet")

    # Exact function names used inside train_asset (from the current train.py)
    with patch("training.train.train_regime_model") as mock_regime:
        mock_regime.return_value = None
        with patch("training.train.train_transition_model") as mock_trans:
            mock_trans.return_value = None
            success = train_asset("XAUUSD", force=force)
            assert success is True


def test_deprecated_wrapper_warning():
    with pytest.raises(ImportError):
        from training.train import train  # old wrapper
