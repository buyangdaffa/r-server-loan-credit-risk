"""Tests for src/data.py: label derivation, merge, and modeling-frame contract."""

from __future__ import annotations

import pandas as pd
import pytest

from src import config, data as data_mod


def test_derive_is_bad_handles_known_statuses() -> None:
    df = pd.DataFrame(
        {
            config.LOAN_STATUS_COL: [
                "Default",
                "Charged Off",
                "Current",
                "default",
                "  Default  ",
                "",
                None,
            ]
        }
    )
    label = data_mod.derive_is_bad(df)
    assert label.name == config.TARGET
    assert label.iloc[0] == 1
    assert label.iloc[1] == 1
    assert label.iloc[2] == 0
    assert label.iloc[3] == 0
    assert label.iloc[4] == 1
    assert pd.isna(label.iloc[5])
    assert pd.isna(label.iloc[6])


def test_merge_loan_borrower_requires_member_id() -> None:
    loan = pd.DataFrame({"loanId": [1]})
    borrower = pd.DataFrame({"memberId": [1]})
    with pytest.raises(KeyError):
        data_mod.merge_loan_borrower(loan, borrower)


def test_merge_loan_borrower_many_to_one() -> None:
    loan = pd.DataFrame({"loanId": [1, 2], "memberId": [10, 10]})
    borrower = pd.DataFrame({"memberId": [10], "income": [50000]})
    out = data_mod.merge_loan_borrower(loan, borrower)
    assert len(out) == 2
    assert set(out.columns) == {"loanId", "memberId", "income"}


def test_build_modeling_frame_drops_id_and_label_source() -> None:
    df = data_mod.build_modeling_frame()
    assert config.TARGET in df.columns
    for col in config.DROP_COLS:
        assert col not in df.columns
    assert df[config.TARGET].between(0, 1).all()
    assert df[config.TARGET].sum() > 0
