"""
Backward-compatible entry point for training.

This wrapper keeps the original import surface used by the tests and by older
automation scripts, while delegating the actual implementation to
training.train.
"""

from __future__ import annotations

import training.train as _train

# Mirror the mutable configuration so external code can patch it.
DATA_DIR = _train.DATA_DIR
MODEL_DIR = _train.MODEL_DIR
FEATURE_COLUMNS = _train.FEATURE_COLUMNS

compute_regime_thresholds = _train.compute_regime_thresholds
assign_regime = _train.assign_regime
create_continuation_target = _train.create_continuation_target
create_transition_target = _train.create_transition_target
train_regime_model = _train.train_regime_model
train_transition_model = _train.train_transition_model


def save_drift_baseline(X, path):
    _train.FEATURE_COLUMNS = FEATURE_COLUMNS
    return _train.save_drift_baseline(X, path)


def train_asset(asset, max_rows=_train.DEFAULT_MAX_ROWS, force=False, force_retrain=False):
    _train.DATA_DIR = DATA_DIR
    _train.MODEL_DIR = MODEL_DIR
    _train.FEATURE_COLUMNS = FEATURE_COLUMNS
    return _train.train_asset(asset, max_rows=max_rows, force=force, force_retrain=force_retrain)


def main():
    _train.DATA_DIR = DATA_DIR
    _train.MODEL_DIR = MODEL_DIR
    _train.FEATURE_COLUMNS = FEATURE_COLUMNS
    _train.main()


if __name__ == "__main__":
    main()
