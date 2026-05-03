"""Load processed CSVs, merge loan + borrower, derive isBad label."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from . import config


def load_raw_processed(processed_dir: Path | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read Loan.csv and Borrower.csv produced by convert_data_to_csv.py."""
    base = processed_dir or config.DATA_PROCESSED
    loan = pd.read_csv(base / "Loan.csv")
    borrower = pd.read_csv(base / "Borrower.csv")
    return loan, borrower


def merge_loan_borrower(loan: pd.DataFrame, borrower: pd.DataFrame) -> pd.DataFrame:
    """Inner-join on memberId. Inner so we never score a loan without borrower data."""
    if "memberId" not in loan.columns or "memberId" not in borrower.columns:
        raise KeyError("memberId required in both loan and borrower frames")
    return loan.merge(borrower, on="memberId", how="inner", validate="many_to_one")


def derive_is_bad(
    df: pd.DataFrame,
    *,
    bad_values: tuple[str, ...] = config.BAD_STATUS_VALUES,
    status_col: str = config.LOAN_STATUS_COL,
) -> pd.Series:
    """Binary label: 1 if loanStatus is in bad_values, else 0.

    Rows with missing/empty loanStatus are treated as unlabeled (NaN) so callers
    can drop them before training.
    """
    if status_col not in df.columns:
        raise KeyError(f"{status_col} not in dataframe")
    status = df[status_col].astype("string").str.strip()
    label = pd.Series(pd.NA, index=df.index, dtype="Int8")
    has_value = status.notna() & (status != "")
    label.loc[has_value & status.isin(bad_values)] = 1
    label.loc[has_value & ~status.isin(bad_values)] = 0
    return label.rename(config.TARGET)


def build_modeling_frame(
    processed_dir: Path | None = None,
) -> pd.DataFrame:
    """Return a DataFrame with features + isBad, ready for splitting/training.

    - Loads processed CSVs.
    - Merges on memberId.
    - Derives isBad.
    - Drops rows with missing label.
    - Drops ID/leakage/date columns.
    """
    loan, borrower = load_raw_processed(processed_dir)
    merged = merge_loan_borrower(loan, borrower)
    merged[config.TARGET] = derive_is_bad(merged)
    merged = merged.dropna(subset=[config.TARGET]).copy()
    merged[config.TARGET] = merged[config.TARGET].astype("int8")
    drop = [c for c in config.DROP_COLS if c in merged.columns]
    return merged.drop(columns=drop)


def load_unlabeled_prod(processed_dir: Path | None = None) -> pd.DataFrame:
    """Load Loan_Prod.csv joined with Borrower_Prod.csv for batch scoring demos."""
    base = processed_dir or config.DATA_PROCESSED
    loan = pd.read_csv(base / "Loan_Prod.csv")
    borrower = pd.read_csv(base / "Borrower_Prod.csv")
    merged = merge_loan_borrower(loan, borrower)
    drop = [c for c in (config.ID_COLS + (config.LOAN_STATUS_COL, "date")) if c in merged.columns]
    return merged.drop(columns=drop)
