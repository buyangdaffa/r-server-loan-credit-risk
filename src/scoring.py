"""Inference helpers used by the CLI and Streamlit app.

Wraps a calibrated model with score-band lookup and per-prediction SHAP
explanations so the Streamlit app can stay focused on UI.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from . import config, modeling


@dataclass
class ScoreBand:
    """Quantile-based score bands derived from training-set predictions."""

    edges: list[float]
    bad_rates: list[float]
    percentiles: list[float]

    def assign(self, prob: float) -> tuple[int, float]:
        idx = int(np.searchsorted(self.edges, prob, side="right"))
        idx = min(idx, len(self.bad_rates) - 1)
        return idx, float(self.bad_rates[idx])

    def to_dict(self) -> dict:
        return {
            "edges": [float(e) for e in self.edges],
            "bad_rates": [float(b) for b in self.bad_rates],
            "percentiles": [float(p) for p in self.percentiles],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ScoreBand":
        return cls(
            edges=[float(e) for e in data["edges"]],
            bad_rates=[float(b) for b in data["bad_rates"]],
            percentiles=[float(p) for p in data["percentiles"]],
        )


def fit_score_bands(
    y_true: np.ndarray,
    y_score: np.ndarray,
    *,
    percentiles: tuple[float, ...] = config.SCORE_BAND_PERCENTILES,
) -> ScoreBand:
    """Compute quantile cutoffs and the observed bad rate above each cutoff."""
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)
    edges = np.quantile(y_score, percentiles).tolist()
    bad_rates: list[float] = []
    for cutoff in edges:
        mask = y_score >= cutoff
        bad_rates.append(float(y_true[mask].mean()) if mask.any() else 0.0)
    return ScoreBand(
        edges=[float(e) for e in edges],
        bad_rates=bad_rates,
        percentiles=[float(p) for p in percentiles],
    )


def save_score_bands(band: ScoreBand, path: Path | str) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as fh:
        json.dump(band.to_dict(), fh, indent=2)


def load_score_bands(path: Path | str) -> ScoreBand:
    p = Path(path)
    with p.open(encoding="utf-8") as fh:
        return ScoreBand.from_dict(json.load(fh))


@dataclass
class Prediction:
    probability: float
    decision: str
    threshold: float
    band_index: int
    band_bad_rate: float
    drivers: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "probability": self.probability,
            "decision": self.decision,
            "threshold": self.threshold,
            "band_index": self.band_index,
            "band_bad_rate": self.band_bad_rate,
            "drivers": list(self.drivers),
        }


def score_one(
    model,
    row: dict | pd.DataFrame,
    *,
    threshold: float,
    band: ScoreBand | None = None,
    explainer=None,
    n_drivers: int = 5,
) -> Prediction:
    """Score a single application and (optionally) attach SHAP drivers."""
    if isinstance(row, dict):
        df = pd.DataFrame([row])
    else:
        df = row.head(1)
    prob = float(modeling.predict_proba(model, df)[0])
    decision = "decline" if prob >= threshold else "approve"
    band_idx = -1
    band_rate = float("nan")
    if band is not None:
        band_idx, band_rate = band.assign(prob)
    drivers: list[dict] = []
    if explainer is not None:
        drivers = [d.to_dict() for d in explainer(model, df, n=n_drivers)]
    return Prediction(
        probability=prob,
        decision=decision,
        threshold=float(threshold),
        band_index=band_idx,
        band_bad_rate=band_rate,
        drivers=drivers,
    )


def score_batch(
    model,
    df: pd.DataFrame,
    *,
    threshold: float,
    band: ScoreBand | None = None,
) -> pd.DataFrame:
    """Return ``df`` augmented with probability, decision, band index/rate."""
    proba = modeling.predict_proba(model, df)
    out = df.copy()
    out["probability"] = proba
    out["decision"] = np.where(proba >= threshold, "decline", "approve")
    if band is not None:
        bands = [band.assign(p) for p in proba]
        out["band_index"] = [b[0] for b in bands]
        out["band_bad_rate"] = [b[1] for b in bands]
    return out
