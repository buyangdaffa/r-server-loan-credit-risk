"""Tests for src/features.py: supervised binning and preprocessor builders."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import config, features


@pytest.fixture
def small_frame() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    n = 200
    return pd.DataFrame(
        {
            "loanAmount": rng.normal(15_000, 5_000, n),
            "interestRate": rng.normal(10, 3, n),
            "monthlyPayment": rng.normal(400, 80, n),
            "annualIncome": rng.normal(60_000, 20_000, n),
            "dtiRatio": rng.normal(20, 5, n),
            "lengthCreditHistory": rng.integers(1, 30, n),
            "numTotalCreditLines": rng.integers(1, 30, n),
            "numOpenCreditLines": rng.integers(0, 20, n).astype(float),
            "numOpenCreditLines1Year": rng.integers(0, 10, n),
            "revolvingBalance": rng.integers(0, 50_000, n),
            "revolvingUtilizationRate": rng.uniform(0, 100, n),
            "numDerogatoryRec": rng.integers(0, 5, n),
            "numDelinquency2Years": rng.integers(0, 5, n),
            "numChargeoff1year": rng.integers(0, 3, n),
            "numInquiries6Mon": rng.integers(0, 5, n),
            "purpose": rng.choice(["debtconsolidation", "homeimprovement"], n),
            "term": rng.choice(["36 months", "60 months"], n),
            "grade": rng.choice(["A1", "B2", "C3"], n),
            "residentialState": rng.choice(["CA", "TX", "NY"], n),
            "yearsEmployment": rng.choice(["< 1 year", "10+ years"], n),
            "homeOwnership": rng.choice(["rent", "own", "mortgage"], n),
            "isJointApplication": rng.integers(0, 2, n),
            "incomeVerified": rng.integers(0, 2, n),
        }
    )


def test_supervised_binner_fit_transform_shapes(small_frame: pd.DataFrame) -> None:
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, len(small_frame))
    cols = list(config.NUMERIC_FEATURES)
    binner = features.SupervisedBinner(max_bins=4)
    out = binner.fit(small_frame[cols], y).transform(small_frame[cols])
    assert out.shape == (len(small_frame), len(cols))
    assert out.dtype.kind in {"i", "u"}
    edges_dict = binner.to_dict()
    assert set(edges_dict) == set(cols)
    for col, edges in edges_dict.items():
        assert isinstance(edges, list)


def test_supervised_binner_requires_y(small_frame: pd.DataFrame) -> None:
    with pytest.raises(ValueError):
        features.SupervisedBinner().fit(small_frame[list(config.NUMERIC_FEATURES)])


def test_lr_preprocessor_runs(small_frame: pd.DataFrame) -> None:
    y = np.random.default_rng(2).integers(0, 2, len(small_frame))
    pre = features.make_lr_preprocessor()
    X = pre.fit_transform(small_frame, y)
    assert X.shape[0] == len(small_frame)
    assert X.shape[1] > len(config.NUMERIC_FEATURES)
    names = features.get_lr_feature_names(pre)
    assert len(names) == X.shape[1]


def test_gbm_preprocessor_runs(small_frame: pd.DataFrame) -> None:
    pre = features.make_gbm_preprocessor()
    X = pre.fit_transform(small_frame)
    assert X.shape[0] == len(small_frame)
    expected_cols = (
        len(config.NUMERIC_FEATURES) + len(config.CATEGORICAL_FEATURES) + len(config.BINARY_FEATURES)
    )
    assert X.shape[1] == expected_cols


def test_export_bins_writes_json(tmp_path, small_frame: pd.DataFrame) -> None:
    y = np.random.default_rng(3).integers(0, 2, len(small_frame))
    binner = features.SupervisedBinner()
    binner.fit(small_frame[list(config.NUMERIC_FEATURES)], y)
    out = tmp_path / "bins.json"
    features.export_bins(binner, out)
    assert out.is_file()
    assert out.stat().st_size > 0
