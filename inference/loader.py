"""
Model loading helpers for CrossStrux v2.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import joblib


MODEL_ROOT = os.environ.get("MODEL_ROOT", "models")


@dataclass
class AssetBundle:
    asset: str
    metadata: Dict[str, Any]
    models: Dict[str, Any]
    transition_model: Optional[Any]
    baseline: Dict[str, Any]


_CACHE: Dict[str, AssetBundle] = {}


def load_asset_bundle(asset: str) -> AssetBundle:
    """
    Load metadata, regime models, transition model, and drift baselines.
    """
    if asset in _CACHE:
        return _CACHE[asset]

    base = os.path.join(MODEL_ROOT, asset)
    metadata_path = os.path.join(base, "metadata.json")

    if not os.path.exists(metadata_path):
        raise FileNotFoundError(
            f"Model metadata not found for asset '{asset}'. Expected {metadata_path}"
        )

    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    models = {}
    for regime in ["low", "mid", "high"]:
        path = os.path.join(base, f"{regime}_model.pkl")
        if os.path.exists(path):
            models[regime] = joblib.load(path)

    transition_path = os.path.join(base, "transition_model.pkl")
    transition_model = joblib.load(transition_path) if os.path.exists(transition_path) else None

    baseline = {}
    for regime in ["low", "mid", "high"]:
        path = os.path.join(base, f"drift_baseline_{regime}.json")
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                baseline[regime] = json.load(f)

    bundle = AssetBundle(
        asset=asset,
        metadata=metadata,
        models=models,
        transition_model=transition_model,
        baseline=baseline,
    )
    _CACHE[asset] = bundle
    return bundle


def loaded_assets() -> list[str]:
    return sorted(_CACHE.keys())
