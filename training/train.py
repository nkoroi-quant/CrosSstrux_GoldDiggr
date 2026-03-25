# training/train.py - HistGradientBoosting + model registry + git hash

import argparse
import json
import logging
import os
import subprocess
from typing import Dict

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from core.features.feature_pipeline import build_features, get_extended_feature_columns
from core.structure.impulse import detect_impulse
from core.structure.balance import detect_balance
from core.structure.volatility import volatility_expansion
from core.regimes.regime_classifier import classify_regime
from core.regimes.stability import regime_stability
from utils.parquet_compat import install as install_parquet_compat
from config.settings import settings

install_parquet_compat()
logger = logging.getLogger(__name__)

# ... (keep all original functions: compute_regime_thresholds, assign_regime, create_*_target etc.)

def train_regime_model(X: pd.DataFrame, y: pd.Series):
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", HistGradientBoostingClassifier(
            max_iter=200,
            learning_rate=0.1,
            max_depth=8,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.1
        ))
    ])
    model.fit(X, y)
    return model

# train_transition_model same with HistGradientBoostingClassifier

def save_metadata(asset_dir: str, git_hash: str):
    metadata = {
        "version": "3.1",
        "git_hash": git_hash,
        "trained_at": pd.Timestamp.utcnow().isoformat(),
        "feature_columns": get_extended_feature_columns(),
        "regimes": ["low", "mid", "high"]
    }
    with open(os.path.join(asset_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"Model registry saved with git hash {git_hash}")

def train_asset(...):  # full original signature kept
    # ... existing data loading + structure + features
    # train with new model
    # save models
    git_hash = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()
    save_metadata(asset_dir, git_hash)
    # save drift baselines
    # return True