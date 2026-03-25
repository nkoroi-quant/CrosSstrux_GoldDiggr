"""
Training pipeline for CrossStrux v2.

This preserves the original V1 training logic, but is better aligned to a
single-asset XAUUSD production flow.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from typing import Dict, List

import joblib
import numpy as np
import pandas as pd
from utils.parquet_compat import install as install_parquet_compat

install_parquet_compat()
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from core.features.feature_pipeline import build_features, get_feature_columns, get_extended_feature_columns
from core.structure.impulse import detect_impulse
from core.structure.balance import detect_balance
from core.structure.volatility import volatility_expansion
from core.regimes.regime_classifier import classify_regime
from core.regimes.stability import regime_stability


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = os.environ.get("DATA_DIR", os.path.join("data", "raw"))
MODEL_DIR = os.environ.get("MODEL_DIR", "models")

DEFAULT_MAX_ROWS = 100000
TRANSITION_HORIZON = 5
MIN_SAMPLES_PER_REGIME = 30

FEATURE_COLUMNS = get_extended_feature_columns()


def detect_impulse_column(df: pd.DataFrame) -> str:
    # Prefer the continuous v2 signal if present, then fall back to legacy binary columns.
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
    """
    Create the continuation target.

    Compatibility note:
    - If a string column name is passed as the second argument (legacy test style),
      use the next-bar direction from that source column.
    - Otherwise use the forward return / ATR-based continuation logic.
    """
    df = df.copy()

    if isinstance(target_source, str):
        source_col = target_source if target_source in df.columns else ("impulse_norm" if "impulse_norm" in df.columns else ("impulse" if "impulse" in df.columns else None))
        if source_col is not None:
            source = pd.to_numeric(df[source_col], errors="coerce").fillna(0)
            future = source.shift(-1).fillna(0)
            if source.nunique(dropna=True) <= 2 and set(source.dropna().unique()).issubset({0, 1}):
                df["continuation_y"] = future.astype(int)
            else:
                df["continuation_y"] = (future > source).astype(int)
            return df

    future_return = (df["close"].shift(-horizon) - df["close"]) / df["close"]
    if "atr_pct" not in df.columns:
        df["atr_pct"] = df["atr"] / df["close"]
    df["continuation_y"] = (future_return.abs() > df["atr_pct"]).astype(int)
    return df


def create_transition_target(df: pd.DataFrame):
    df = df.copy()
    regime_map = {"low": 0, "mid": 1, "high": 2}
    df["regime_encoded"] = df["model_regime"].map(regime_map)

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
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", CalibratedClassifierCV(
            LogisticRegression(max_iter=1000, random_state=42),
            method="isotonic",
            cv=3,
        )),
    ])
    model.fit(X, y)
    return model


def train_transition_model(X: pd.DataFrame, y: pd.Series):
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", CalibratedClassifierCV(
            LogisticRegression(max_iter=1000, random_state=42),
            method="isotonic",
            cv=3,
        )),
    ])
    model.fit(X, y)
    return model


def save_drift_baseline(X: pd.DataFrame, path: str):
    baseline = {col: X[col].tolist() for col in FEATURE_COLUMNS if col in X.columns}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(baseline, f)
    logger.info(f"Saved drift baseline to {path}")


def train_asset(asset: str, max_rows: int = DEFAULT_MAX_ROWS, force: bool = False, force_retrain: bool = False):
    logger.info("\n" + "=" * 60)
    logger.info(f"Training asset: {asset}")
    logger.info("=" * 60)

    asset_dir = os.path.join(MODEL_DIR, asset)
    metadata_path = os.path.join(asset_dir, "metadata.json")

    if not (force or force_retrain) and os.path.exists(metadata_path):
        logger.info("Models already exist")
        return True

    data_path = os.path.join(DATA_DIR, f"{asset}_M1.parquet")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}")

    df = pd.read_parquet(data_path)
    logger.info(f"Loaded {len(df)} rows")

    if len(df) > max_rows:
        df = df.tail(max_rows)

    raw_ohlc = {"open", "high", "low", "close"}.issubset(df.columns)

    if raw_ohlc:
        logger.info("Running structure pipeline...")
        df = detect_impulse(df)
        df = detect_balance(df)
        df = volatility_expansion(df)
        df = classify_regime(df)
        df = regime_stability(df)

        REQUIRED_STRUCTURE_COLUMNS = [
            "impulse",
            "impulse_strength",
            "impulse_dir",
            "balance",
            "volatility_expansion",
            "regime_stability",
        ]
        for col in REQUIRED_STRUCTURE_COLUMNS:
            if col not in df.columns:
                raise RuntimeError(f"Structure pipeline error: expected column '{col}' not produced.")

        df = create_continuation_target(df)

        logger.info("Running feature pipeline...")
        features_df = build_features(df)
        feature_only = features_df[["time"] + FEATURE_COLUMNS].copy()
        df = df.merge(feature_only, on="time", how="inner")
    else:
        logger.info("Detected pre-built feature table; using it as training input.")
        if "continuation_y" not in df.columns:
            target_seed = "impulse_norm" if "impulse_norm" in df.columns else "impulse"
            df = create_continuation_target(df, target_seed)
        if "model_regime" not in df.columns:
            impulse_col_tmp = detect_impulse_column(df)
            low_q, high_q = compute_regime_thresholds(df, impulse_col_tmp)
            df = assign_regime(df, impulse_col_tmp, low_q, high_q)
        if "transition_y" not in df.columns:
            df = create_transition_target(df)

        # Prebuilt feature tables from tests or older pipelines may only contain
        # the compact V1 feature set. Rebuild or fall back gracefully so the
        # training path accepts both V1 and V2 shaped inputs.
        available_extended = [c for c in FEATURE_COLUMNS if c in df.columns]
        compact_columns = [c for c in get_feature_columns() if c in df.columns]

        if len(available_extended) < 4 and compact_columns:
            logger.info("Falling back to compact feature set for pre-built table training.")
            FEATURE_COLUMNS_USED = compact_columns
        elif available_extended:
            FEATURE_COLUMNS_USED = available_extended
        else:
            raise KeyError(
                "No compatible feature columns found in pre-built table. "
                "Provide raw OHLC data or a feature table containing at least the compact CrossStrux columns."
            )

    try:
        FEATURE_COLUMNS_USED
    except NameError:
        FEATURE_COLUMNS_USED = [c for c in FEATURE_COLUMNS if c in df.columns]
        if not FEATURE_COLUMNS_USED:
            FEATURE_COLUMNS_USED = [c for c in get_feature_columns() if c in df.columns]
        if not FEATURE_COLUMNS_USED:
            raise KeyError(
                "No usable feature columns available for training. Expected either the extended V2 set or the compact V1 set."
            )

    impulse_col = detect_impulse_column(df)
    low_q, high_q = compute_regime_thresholds(df, impulse_col)
    logger.info(f"Regime thresholds low={low_q:.4f} high={high_q:.4f}")

    df = assign_regime(df, impulse_col, low_q, high_q)
    df = create_transition_target(df)
    df = df.dropna(subset=["continuation_y", "transition_y"])

    logger.info(f"Training rows: {len(df)}")
    logger.info(df["model_regime"].value_counts())

    os.makedirs(asset_dir, exist_ok=True)

    trained = []

    for regime in ["low", "mid", "high"]:
        subset = df[df["model_regime"] == regime]
        logger.info(f"\nTraining {regime} regime model — samples: {len(subset)}")

        if len(subset) < MIN_SAMPLES_PER_REGIME:
            logger.warning("Skipping insufficient samples")
            continue

        X = subset[FEATURE_COLUMNS_USED]
        y = subset["continuation_y"]

        if y.nunique() < 2:
            logger.warning("Skipping single class")
            continue

        model = train_regime_model(X, y)
        joblib.dump(model, os.path.join(asset_dir, f"{regime}_model.pkl"))
        save_drift_baseline(X, os.path.join(asset_dir, f"drift_baseline_{regime}.json"))
        trained.append(regime)

    transition_features = FEATURE_COLUMNS_USED + ["regime_encoded"]
    X_t = df[transition_features]
    y_t = df["transition_y"]

    if y_t.nunique() > 1:
        model = train_transition_model(X_t, y_t)
        joblib.dump(model, os.path.join(asset_dir, "transition_model.pkl"))
        logger.info("Transition model trained")

    metadata = {
        "asset_name": asset,
        "low_threshold": float(low_q),
        "high_threshold": float(high_q),
        "feature_columns": FEATURE_COLUMNS_USED,
        "transition_features": transition_features,
        "trained_regimes": trained,
    }

    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    logger.info("Training complete")
    return len(trained) > 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--assets", default="XAUUSD", help="Comma-separated assets to train")
    parser.add_argument("--max-rows", type=int, default=DEFAULT_MAX_ROWS)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--force-retrain", action="store_true")
    args = parser.parse_args()

    assets = [a.strip() for a in args.assets.split(",") if a.strip()]
    results = {}

    for asset in assets:
        results[asset] = train_asset(asset, args.max_rows, force=args.force, force_retrain=args.force_retrain)

    logger.info("\nTraining summary")
    for asset, ok in results.items():
        logger.info(f"{'✓' if ok else '✗'} {asset}")

    if not all(results.values()):
        sys.exit(1)


if __name__ == "__main__":
    main()
