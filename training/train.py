"""
Training pipeline for CrossStrux v3.1 - Optimized with HistGradientBoosting + model registry.

Preserves original V1/V2 logic while switching to faster, more accurate HistGradientBoostingClassifier
and adding Git hash metadata for reproducibility.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
from typing import Dict, List

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from utils.parquet_compat import install as install_parquet_compat

install_parquet_compat()

from core.features.feature_pipeline import build_features, get_extended_feature_columns
from core.structure.impulse import detect_impulse
from core.structure.balance import detect_balance
from core.structure.volatility import volatility_expansion
from core.regimes.regime_classifier import classify_regime
from core.regimes.stability import regime_stability
from config.settings import settings

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = os.environ.get("DATA_DIR", os.path.join("data", "raw"))
MODEL_DIR = os.environ.get("MODEL_DIR", settings.MODEL_ROOT)

DEFAULT_MAX_ROWS = 100000
TRANSITION_HORIZON = 5

FEATURE_COLUMNS = get_extended_feature_columns()


def detect_impulse_column(df: pd.DataFrame) -> str:
    preferred = ["impulse_norm", "impulse_strength", "impulse", "impulse_dir"]
    for col in preferred:
        if col in df.columns:
            return col
    candidates = [c for c in df.columns if "impulse" in c]
    if not candidates:
        raise ValueError("No impulse column found")
    return candidates[0]


def compute_regime_thresholds(df: pd.DataFrame, impulse_col: str):
    low_q = df[impulse_col].quantile(0.4)
    high_q = df[impulse_col].quantile(0.8)
    return float(low_q), float(high_q)


def assign_regime(df: pd.DataFrame, impulse_col: str, low_q: float, high_q: float):
    df = df.copy()
    df["model_regime"] = np.where(
        df[impulse_col] <= low_q,
        "low",
        np.where(df[impulse_col] <= high_q, "mid", "high"),
    )
    return df


def create_continuation_target(df: pd.DataFrame, target_source=None, horizon: int = 10):
    df = df.copy()

    if isinstance(target_source, str):
        source_col = (
            target_source
            if target_source in df.columns
            else (
                "impulse_norm"
                if "impulse_norm" in df.columns
                else ("impulse" if "impulse" in df.columns else None)
            )
        )
        if source_col is not None:
            source = pd.to_numeric(df[source_col], errors="coerce").fillna(0)
            future = source.shift(-1).fillna(0)
            if source.nunique(dropna=True) <= 2 and set(source.dropna().unique()).issubset({0, 1}):
                df["continuation_y"] = future.astype(int)
            else:
                df["continuation_y"] = (future > source).astype(int)
            return df

    future_return = (df["close"].shift(-horizon) - df["close"]) / df["close"]
    if "atr" in df.columns:
        df["atr_pct"] = df["atr"] / df["close"]
        df["continuation_y"] = (future_return.abs() > df["atr_pct"]).astype(int)
    else:
        df["continuation_y"] = (future_return > 0).astype(int)
    return df


def create_transition_target(df: pd.DataFrame):
    df = df.copy()
    regime_map = {"low": 0, "mid": 1, "high": 2}
    df["regime_encoded"] = df["model_regime"].map(regime_map).fillna(1)

    regimes = df["regime_encoded"].values
    targets = []

    for i in range(len(regimes)):
        future = regimes[i + 1 : i + 1 + TRANSITION_HORIZON]
        if len(future) == 0:
            targets.append(0)
        elif any(r != regimes[i] for r in future):
            targets.append(1)
        else:
            targets.append(0)

    df["transition_y"] = targets
    return df


def train_regime_model(X: pd.DataFrame, y: pd.Series):
    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "clf",
                HistGradientBoostingClassifier(
                    max_iter=400,
                    learning_rate=0.08,
                    max_depth=9,
                    random_state=42,
                    early_stopping=True,
                    validation_fraction=0.15,
                    n_iter_no_change=15,
                    verbose=0,
                ),
            ),
        ]
    )
    model.fit(X, y)
    return model


def train_transition_model(X: pd.DataFrame, y: pd.Series):
    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "clf",
                HistGradientBoostingClassifier(
                    max_iter=300,
                    learning_rate=0.1,
                    max_depth=7,
                    random_state=42,
                    early_stopping=True,
                    validation_fraction=0.1,
                ),
            ),
        ]
    )
    model.fit(X, y)
    return model


def save_drift_baseline(X: pd.DataFrame, path: str):
    baseline = {col: X[col].tolist() for col in FEATURE_COLUMNS if col in X.columns}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(baseline, f)
    logger.info(f"Saved drift baseline to {path}")


def save_metadata(asset_dir: str, git_hash: str):
    metadata = {
        "version": "3.1",
        "git_hash": git_hash,
        "trained_at": pd.Timestamp.utcnow().isoformat(),
        "feature_columns": FEATURE_COLUMNS,
        "regimes": ["low", "mid", "high"],
        "model_type": "HistGradientBoostingClassifier",
    }
    with open(os.path.join(asset_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"Model registry metadata saved with git hash {git_hash[:8]}")


def train_asset(
    asset: str, max_rows: int = DEFAULT_MAX_ROWS, force: bool = False, force_retrain: bool = False
):
    logger.info("\n" + "=" * 60)
    logger.info(f"Training asset: {asset}")
    logger.info("=" * 60)

    asset_dir = os.path.join(MODEL_DIR, asset)
    os.makedirs(asset_dir, exist_ok=True)
    metadata_path = os.path.join(asset_dir, "metadata.json")

    if not (force or force_retrain) and os.path.exists(metadata_path):
        logger.info("Models already exist for this asset. Use --force-retrain to override.")
        return True

    data_path = os.path.join(DATA_DIR, f"{asset}_M1.parquet")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}")

    df = pd.read_parquet(data_path, use_threads=True)
    logger.info(f"Loaded {len(df):,} rows from {data_path}")

    if len(df) > max_rows:
        df = df.tail(max_rows).reset_index(drop=True)

    raw_ohlc = {"open", "high", "low", "close"}.issubset(df.columns)

    if raw_ohlc:
        logger.info("Running full structure + feature pipeline...")
        df = detect_impulse(df)
        df = detect_balance(df)
        df = volatility_expansion(df)
        df = classify_regime(df)
        df = regime_stability(df)

        df = create_continuation_target(df)

        # Build features and merge safely
        features_df = build_features(df)
        feature_cols = [c for c in FEATURE_COLUMNS if c in features_df.columns]
        feature_only = features_df[["time"] + feature_cols].copy()
        df = df.merge(feature_only, on="time", how="left")

        # Defensive: ensure model_regime exists
        if "model_regime" not in df.columns:
            logger.warning("'model_regime' missing after pipeline. Computing fallback thresholds.")
            impulse_col = detect_impulse_column(df)
            low_q, high_q = compute_regime_thresholds(df, impulse_col)
            df = assign_regime(df, impulse_col, low_q, high_q)

        # Defensive: ensure all FEATURE_COLUMNS exist (fill missing with 0 or median)
        missing_features = [c for c in FEATURE_COLUMNS if c not in df.columns]
        if missing_features:
            logger.warning(f"Missing features: {missing_features}. Filling with 0.")
            for c in missing_features:
                df[c] = 0.0
    else:
        logger.info("Using pre-built feature table.")
        if "continuation_y" not in df.columns:
            target_seed = "impulse_norm" if "impulse_norm" in df.columns else "impulse"
            df = create_continuation_target(df, target_seed)
        if "model_regime" not in df.columns:
            impulse_col_tmp = detect_impulse_column(df)
            low_q, high_q = compute_regime_thresholds(df, impulse_col_tmp)
            df = assign_regime(df, impulse_col_tmp, low_q, high_q)
        if "transition_y" not in df.columns:
            df = create_transition_target(df)

    # Regime model
    regime_mask = df["model_regime"].isin(["low", "mid", "high"])
    X_reg = df.loc[regime_mask, FEATURE_COLUMNS]
    y_reg = df.loc[regime_mask, "model_regime"]

    if len(y_reg.unique()) < 2:
        logger.warning("Not enough regime diversity - skipping regime model")
    else:
        regime_model = train_regime_model(X_reg, y_reg)
        joblib.dump(regime_model, os.path.join(asset_dir, "regime_model.pkl"))
        logger.info("Regime model trained and saved")

    # Continuation model
    if "continuation_y" in df.columns:
        X_cont = df[FEATURE_COLUMNS]
        y_cont = df["continuation_y"]
        cont_model = train_regime_model(X_cont, y_cont)
        joblib.dump(cont_model, os.path.join(asset_dir, "continuation_model.pkl"))
        logger.info("Continuation model trained")

    # Transition model
    if "transition_y" in df.columns:
        X_trans = df[FEATURE_COLUMNS]
        y_trans = df["transition_y"]
        trans_model = train_transition_model(X_trans, y_trans)
        joblib.dump(trans_model, os.path.join(asset_dir, "transition_model.pkl"))
        logger.info("Transition model trained")

    # Drift baseline (last 10k samples)
    save_drift_baseline(df.tail(10000), os.path.join(asset_dir, "drift_baseline.json"))

    # Metadata with git hash
    try:
        git_hash = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()
    except Exception:
        git_hash = "unknown"

    save_metadata(asset_dir, git_hash)

    logger.info(f"✅ Training completed successfully for {asset}")
    return True


def main():
    parser = argparse.ArgumentParser(description="CrossStrux GoldDiggr Training Pipeline v3.1")
    parser.add_argument("--assets", type=str, default="XAUUSD", help="Comma-separated assets")
    parser.add_argument("--max-rows", type=int, default=DEFAULT_MAX_ROWS)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--force-retrain", action="store_true")
    args = parser.parse_args()

    assets = [a.strip() for a in args.assets.split(",")]
    for asset in assets:
        train_asset(
            asset, max_rows=args.max_rows, force=args.force, force_retrain=args.force_retrain
        )


if __name__ == "__main__":
    main()
