"""Tests for src/scoring.py: ScoreBand, score_one, score_batch."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src import scoring


class _DummyModel:
    """Predicts probability proportional to an input column for deterministic tests."""

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        p = np.clip(X["score_input"].to_numpy().astype(float), 0.0, 1.0)
        return np.column_stack([1 - p, p])


def test_score_band_assign_monotonic_index() -> None:
    band = scoring.ScoreBand(
        edges=[0.1, 0.3, 0.6, 0.9],
        bad_rates=[0.05, 0.2, 0.4, 0.7],
        percentiles=[0.25, 0.5, 0.75, 1.0],
    )
    assert band.assign(0.05)[0] == 0
    assert band.assign(0.2)[0] == 1
    assert band.assign(0.5)[0] == 2
    assert band.assign(0.95)[0] == 3
    assert band.assign(1.5)[0] == 3


def test_score_band_round_trip(tmp_path) -> None:
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 1000)
    p = rng.random(1000)
    band = scoring.fit_score_bands(y, p)
    out_path = tmp_path / "bands.json"
    scoring.save_score_bands(band, out_path)
    loaded = scoring.load_score_bands(out_path)
    assert loaded.edges == band.edges
    assert loaded.bad_rates == band.bad_rates


def test_score_one_decision_below_threshold() -> None:
    model = _DummyModel()
    pred = scoring.score_one(model, {"score_input": 0.2}, threshold=0.5)
    assert pred.probability == 0.2
    assert pred.decision == "approve"


def test_score_one_decision_above_threshold() -> None:
    model = _DummyModel()
    pred = scoring.score_one(model, {"score_input": 0.8}, threshold=0.5)
    assert pred.decision == "decline"
    assert pred.threshold == 0.5
    assert pred.drivers == []


def test_score_batch_columns_and_shape() -> None:
    model = _DummyModel()
    df = pd.DataFrame({"score_input": [0.1, 0.5, 0.9]})
    band = scoring.ScoreBand(edges=[0.0, 0.5, 1.0], bad_rates=[0.1, 0.4, 0.8], percentiles=[0.33, 0.66, 1.0])
    out = scoring.score_batch(model, df, threshold=0.5, band=band)
    assert list(out.columns) == ["score_input", "probability", "decision", "band_index", "band_bad_rate"]
    assert out["decision"].tolist() == ["approve", "decline", "decline"]
    assert len(out) == 3
