"""Model training, calibration, and artifact persistence.

Two estimators are produced:

- ``train_lr``: ``LogisticRegression`` on the LR preprocessor pipeline. The
  binned + one-hot features keep the model interpretable, in the spirit of
  the upstream credit-scoring sample.
- ``train_lgbm``: ``LightGBMClassifier`` on the GBM preprocessor pipeline.

Both are wrapped with isotonic ``CalibratedClassifierCV`` so the published
probabilities are well-calibrated and the operational thresholds can be
chosen meaningfully.

Artifacts are written under ``data/artifacts/``:

- ``lr_model.joblib`` / ``lgbm_model.joblib`` calibrated pipelines.
- ``bins.json`` LR supervised bin edges.
- ``metadata.json`` git SHA, seed, library versions, training row counts.
"""

from __future__ import annotations

import json
import platform
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from . import config, features


@dataclass
class TrainSplit:
    """Container holding the train/test split for both models."""

    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series


def make_split(
    df: pd.DataFrame,
    *,
    target: str = config.TARGET,
    test_size: float = config.TEST_SIZE,
    random_state: int = config.RANDOM_STATE,
) -> TrainSplit:
    """Stratified train/test split."""
    y = df[target]
    X = df.drop(columns=[target])
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )
    return TrainSplit(X_train, X_test, y_train, y_test)


def _calibrate(estimator: Pipeline, *, method: str = "isotonic", cv: int = 3) -> CalibratedClassifierCV:
    return CalibratedClassifierCV(estimator=estimator, method=method, cv=cv)


def train_lr(split: TrainSplit) -> CalibratedClassifierCV:
    """Logistic regression on binned features, isotonic-calibrated."""
    base = Pipeline(
        steps=[
            ("preprocess", features.make_lr_preprocessor()),
            (
                "model",
                LogisticRegression(
                    max_iter=2000,
                    class_weight="balanced",
                    solver="lbfgs",
                    n_jobs=None,
                    random_state=config.RANDOM_STATE,
                ),
            ),
        ]
    )
    calibrated = _calibrate(base)
    calibrated.fit(split.X_train, split.y_train)
    return calibrated


def train_lgbm(split: TrainSplit) -> CalibratedClassifierCV:
    """LightGBM classifier on raw features, isotonic-calibrated."""
    base = Pipeline(
        steps=[
            ("preprocess", features.make_gbm_preprocessor()),
            (
                "model",
                lgb.LGBMClassifier(
                    n_estimators=400,
                    learning_rate=0.05,
                    num_leaves=31,
                    max_depth=-1,
                    min_child_samples=50,
                    subsample=0.9,
                    subsample_freq=1,
                    colsample_bytree=0.9,
                    reg_lambda=1.0,
                    is_unbalance=True,
                    random_state=config.RANDOM_STATE,
                    n_jobs=-1,
                    verbose=-1,
                ),
            ),
        ]
    )
    calibrated = _calibrate(base)
    calibrated.fit(split.X_train, split.y_train)
    return calibrated


def extract_lr_binner(model: CalibratedClassifierCV) -> features.SupervisedBinner | None:
    """Reach into a calibrated LR pipeline to fetch the SupervisedBinner.

    ``CalibratedClassifierCV`` keeps a list of fitted clones in
    ``calibrated_classifiers_``; we grab the binner from the first clone (all
    clones are fit on the same numeric column set, so their bin edges differ
    only in the cv folds; the first is representative for export).
    """
    if not getattr(model, "calibrated_classifiers_", None):
        return None
    pipe = model.calibrated_classifiers_[0].estimator
    if not isinstance(pipe, Pipeline):
        return None
    pre = pipe.named_steps.get("preprocess")
    if pre is None:
        return None
    transformers = getattr(pre, "named_transformers_", {})
    num_pipe = transformers.get("num")
    if num_pipe is None:
        return None
    binner = num_pipe.named_steps.get("binner")
    return binner if isinstance(binner, features.SupervisedBinner) else None


def _git_sha() -> str | None:
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(config.ROOT))
            .decode()
            .strip()
        )
    except Exception:
        return None


def write_metadata(
    split: TrainSplit,
    *,
    extra: dict | None = None,
    path: Path | None = None,
) -> Path:
    """Persist run metadata for reproducibility."""
    target_path = path or (config.DATA_ARTIFACTS / "metadata.json")
    target_path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_sha": _git_sha(),
        "random_state": config.RANDOM_STATE,
        "test_size": config.TEST_SIZE,
        "n_train": int(len(split.X_train)),
        "n_test": int(len(split.X_test)),
        "train_pos_rate": float(split.y_train.mean()),
        "test_pos_rate": float(split.y_test.mean()),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "feature_schema": {
            "numeric": list(config.NUMERIC_FEATURES),
            "categorical": list(config.CATEGORICAL_FEATURES),
            "binary": list(config.BINARY_FEATURES),
        },
    }
    try:
        import sklearn

        payload["sklearn_version"] = sklearn.__version__
    except Exception:
        pass
    try:
        payload["lightgbm_version"] = lgb.__version__
    except Exception:
        pass
    if extra:
        payload.update(extra)
    with target_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, default=str)
    return target_path


def save_model(model, name: str, *, artifacts_dir: Path | None = None) -> Path:
    target = (artifacts_dir or config.DATA_ARTIFACTS) / f"{name}.joblib"
    target.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, target)
    return target


def load_model(name: str, *, artifacts_dir: Path | None = None):
    target = (artifacts_dir or config.DATA_ARTIFACTS) / f"{name}.joblib"
    return joblib.load(target)


def predict_proba(model, X: pd.DataFrame) -> np.ndarray:
    """Probability of the positive (bad-loan) class."""
    proba = model.predict_proba(X)
    return proba[:, 1]
