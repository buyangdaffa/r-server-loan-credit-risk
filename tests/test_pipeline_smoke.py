"""End-to-end smoke test on a tiny seeded subsample.

Trains LR + LightGBM on a 1k slice, runs evaluation + scoring, and asserts the
output contracts. Intended to keep CI fast (well under a minute) while catching
schema/pipeline regressions.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src import config, data as data_mod, evaluation, modeling, scoring


@pytest.mark.smoke
def test_pipeline_smoke(tmp_path: Path) -> None:
    df = data_mod.build_modeling_frame()
    sample = df.sample(n=1000, random_state=config.RANDOM_STATE).reset_index(drop=True)
    split = modeling.make_split(sample)
    assert split.X_train.shape[0] + split.X_test.shape[0] == 1000

    lr = modeling.train_lr(split)
    proba_lr = modeling.predict_proba(lr, split.X_test)
    metrics_lr = evaluation.evaluate(split.y_test, proba_lr)
    assert metrics_lr.auc > 0.6
    assert metrics_lr.ks > 0.2

    lgbm = modeling.train_lgbm(split)
    proba_lgbm = modeling.predict_proba(lgbm, split.X_test)
    metrics_lgbm = evaluation.evaluate(split.y_test, proba_lgbm)
    assert metrics_lgbm.auc > 0.6

    band = scoring.fit_score_bands(split.y_train, modeling.predict_proba(lgbm, split.X_train))
    assert len(band.edges) == len(band.bad_rates) == len(config.SCORE_BAND_PERCENTILES)

    scored = scoring.score_batch(lgbm, split.X_test, threshold=metrics_lgbm.threshold, band=band)
    assert {"probability", "decision", "band_index", "band_bad_rate"}.issubset(scored.columns)
    assert scored["probability"].between(0.0, 1.0).all()

    one = scoring.score_one(lgbm, split.X_test.iloc[0].to_dict(), threshold=metrics_lgbm.threshold, band=band)
    assert 0.0 <= one.probability <= 1.0
    assert one.decision in {"approve", "decline"}

    json_path, md_path = evaluation.write_report({"lr": metrics_lr, "lgbm": metrics_lgbm}, tmp_path)
    assert json_path.is_file() and md_path.is_file()
