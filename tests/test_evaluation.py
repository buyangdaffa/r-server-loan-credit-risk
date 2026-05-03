"""Tests for src/evaluation.py: KS, lift, full evaluate contract."""

from __future__ import annotations

import numpy as np

from src import evaluation


def test_ks_perfect_separation() -> None:
    y = np.array([0, 0, 0, 1, 1, 1])
    p = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
    ks, thr = evaluation.ks_statistic(y, p)
    assert ks == 1.0
    assert 0.3 < thr <= 0.9


def test_decile_lift_top_decile_highest() -> None:
    rng = np.random.default_rng(0)
    n = 5000
    y = rng.integers(0, 2, n)
    p = y * 0.5 + rng.random(n) * 0.5
    lifts, bad_rate = evaluation.decile_lift(y, p, n_bins=10)
    assert len(lifts) == 10
    assert lifts[0] >= lifts[-1]
    assert all(0.0 <= b <= 1.0 for b in bad_rate)


def test_evaluate_returns_full_payload() -> None:
    rng = np.random.default_rng(1)
    n = 2000
    y = rng.integers(0, 2, n)
    p = y * 0.3 + rng.random(n) * 0.7
    m = evaluation.evaluate(y, p)
    assert 0.0 <= m.auc <= 1.0
    assert 0.0 <= m.pr_auc <= 1.0
    assert 0.0 <= m.ks <= 1.0
    assert 0.0 <= m.brier <= 1.0
    assert set(m.confusion) == {"tn", "fp", "fn", "tp"}
    assert len(m.lift_by_decile) == 10
    assert "prob_pred" in m.calibration and "prob_true" in m.calibration


def test_write_report_creates_files(tmp_path) -> None:
    rng = np.random.default_rng(2)
    y = rng.integers(0, 2, 500)
    p = rng.random(500)
    m = evaluation.evaluate(y, p)
    json_path, md_path = evaluation.write_report({"toy": m}, tmp_path)
    assert json_path.is_file() and md_path.is_file()
    assert json_path.stat().st_size > 0
    assert "Loan credit risk" in md_path.read_text(encoding="utf-8")
