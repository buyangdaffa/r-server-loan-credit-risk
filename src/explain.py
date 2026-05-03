"""SHAP-based explainability for the LightGBM credit-risk model.

The Streamlit app reads ``top_drivers`` to render a per-prediction explanation;
``shap_values_for_row`` is also available for callers that want full SHAP arrays.

Note: SHAP values are computed against the underlying ``LGBMClassifier`` inside
the calibrated pipeline. Calibration only rescales probabilities monotonically,
so the relative ranking of drivers is preserved.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import shap
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline


@dataclass
class Driver:
    feature: str
    value: object
    shap_value: float
    direction: str

    def to_dict(self) -> dict:
        return {
            "feature": self.feature,
            "value": self.value,
            "shap_value": float(self.shap_value),
            "direction": self.direction,
        }


def _unwrap_lgbm(model: CalibratedClassifierCV) -> tuple[object, Pipeline]:
    """Return the (LGBMClassifier, fitted_preprocessor) inside a calibrated pipeline."""
    if not getattr(model, "calibrated_classifiers_", None):
        raise ValueError("Model is not a fitted CalibratedClassifierCV")
    pipe = model.calibrated_classifiers_[0].estimator
    if not isinstance(pipe, Pipeline):
        raise ValueError("Underlying estimator is not a Pipeline")
    pre = pipe.named_steps["preprocess"]
    booster = pipe.named_steps["model"]
    return booster, pre


def shap_values_for_row(model: CalibratedClassifierCV, row: pd.DataFrame) -> tuple[np.ndarray, list[str], np.ndarray]:
    """Return (shap_values, feature_names, transformed_row) for a single row.

    ``shap_values`` is shape (n_features,) for the positive class.
    """
    booster, pre = _unwrap_lgbm(model)
    Xt = pre.transform(row)
    feat_names = list(pre.get_feature_names_out())
    explainer = shap.TreeExplainer(booster)
    raw = explainer.shap_values(Xt)
    if isinstance(raw, list):
        sv = raw[1]
    else:
        sv = raw
        if sv.ndim == 3:
            sv = sv[..., 1]
    sv = np.asarray(sv).reshape(-1, len(feat_names))[0]
    return sv, feat_names, np.asarray(Xt).reshape(-1, len(feat_names))[0]


def top_drivers(
    model: CalibratedClassifierCV,
    row: pd.DataFrame,
    *,
    n: int = 5,
) -> list[Driver]:
    """Top-N signed drivers for one prediction.

    Drivers are ordered by absolute SHAP value. ``direction`` is "+" when the
    feature pushes probability toward the bad class, "-" otherwise. ``value``
    is the raw input value (from ``row``), not the transformed one, so it
    reads naturally in the UI.
    """
    sv, feat_names, _ = shap_values_for_row(model, row)
    order = np.argsort(np.abs(sv))[::-1][:n]
    drivers: list[Driver] = []
    raw_row = row.iloc[0]
    for idx in order:
        feat = feat_names[idx]
        raw_value = _lookup_raw(raw_row, feat)
        drivers.append(
            Driver(
                feature=feat,
                value=raw_value,
                shap_value=float(sv[idx]),
                direction="+" if sv[idx] >= 0 else "-",
            )
        )
    return drivers


def _lookup_raw(row: pd.Series, transformed_name: str):
    """Map a transformed feature name back to a raw column when possible."""
    if transformed_name in row.index:
        return row[transformed_name]
    base = transformed_name.split("_")[0]
    return row.get(base, np.nan)
