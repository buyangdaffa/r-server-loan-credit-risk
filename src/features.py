"""Feature engineering: imputation, supervised binning, and preprocessor builders.

Two preprocessors are exposed:

- ``make_lr_preprocessor``: numeric features pass through median imputation and
  decision-tree-based supervised binning, then everything is one-hot encoded
  (interpretable baseline for logistic regression).
- ``make_gbm_preprocessor``: numeric features pass through median imputation;
  categoricals are ordinal-encoded (LightGBM handles them natively).

Both preprocessors are sklearn ``Pipeline`` objects, so all parameters are fit
on the training fold only — no leakage from the test fold into encoding,
imputation, or bin edges.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.tree import DecisionTreeClassifier

from . import config


class SupervisedBinner(BaseEstimator, TransformerMixin):
    """Per-column decision-tree binning fit against a binary target.

    For each numeric column we fit a shallow ``DecisionTreeClassifier`` and
    extract its split thresholds. The transform converts each value to its
    bin index (integer). Trees that produce no splits collapse the column to
    a single bin (0).

    Parameters
    ----------
    max_bins:
        Upper bound on the number of bins (= number of leaves) per column.
    min_samples_leaf:
        Minimum fraction (in [0, 0.5]) or count of samples required at a leaf.
    random_state:
        Seed forwarded to the underlying decision trees.
    """

    def __init__(
        self,
        max_bins: int = 6,
        min_samples_leaf: float | int = 0.05,
        random_state: int = config.RANDOM_STATE,
    ) -> None:
        self.max_bins = max_bins
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state

    def fit(self, X: pd.DataFrame | np.ndarray, y=None):
        if y is None:
            raise ValueError("SupervisedBinner requires y for fit")
        X = self._as_frame(X)
        y_arr = np.asarray(y).ravel()
        self.feature_names_in_ = list(X.columns)
        self.bin_edges_: dict[str, np.ndarray] = {}
        for col in self.feature_names_in_:
            values = pd.to_numeric(X[col], errors="coerce")
            mask = values.notna().to_numpy()
            if mask.sum() < 10:
                self.bin_edges_[col] = np.array([], dtype=float)
                continue
            tree = DecisionTreeClassifier(
                max_leaf_nodes=self.max_bins,
                min_samples_leaf=self.min_samples_leaf,
                random_state=self.random_state,
            )
            tree.fit(values[mask].to_numpy().reshape(-1, 1), y_arr[mask])
            thresholds = tree.tree_.threshold[tree.tree_.feature != -2]
            self.bin_edges_[col] = np.sort(np.unique(thresholds))
        return self

    def transform(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        X = self._as_frame(X)
        cols = []
        for col in self.feature_names_in_:
            values = pd.to_numeric(X[col], errors="coerce").to_numpy()
            edges = self.bin_edges_[col]
            if edges.size == 0:
                cols.append(np.zeros(len(values), dtype=np.int16))
                continue
            bins = np.digitize(values, edges, right=False)
            bins = np.where(np.isnan(values), -1, bins).astype(np.int16)
            cols.append(bins)
        return np.column_stack(cols)

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        names = input_features if input_features is not None else self.feature_names_in_
        return np.array([f"{n}_bin" for n in names])

    def to_dict(self) -> dict[str, list[float]]:
        return {col: edges.tolist() for col, edges in self.bin_edges_.items()}

    @staticmethod
    def _as_frame(X) -> pd.DataFrame:
        return X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)


def make_lr_preprocessor(
    numeric: list[str] | None = None,
    categorical: list[str] | None = None,
    binary: list[str] | None = None,
) -> Pipeline:
    """Median imputation + supervised binning + one-hot for LR baseline."""
    numeric = list(numeric or config.NUMERIC_FEATURES)
    categorical = list(categorical or config.CATEGORICAL_FEATURES)
    binary = list(binary or config.BINARY_FEATURES)

    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("binner", SupervisedBinner()),
            (
                "onehot",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )
    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )
    binary_pipe = Pipeline(steps=[("imputer", SimpleImputer(strategy="most_frequent"))])
    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, numeric),
            ("cat", categorical_pipe, categorical),
            ("bin", binary_pipe, binary),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def make_gbm_preprocessor(
    numeric: list[str] | None = None,
    categorical: list[str] | None = None,
    binary: list[str] | None = None,
) -> Pipeline:
    """Median imputation + ordinal-encoded categoricals for LightGBM."""
    numeric = list(numeric or config.NUMERIC_FEATURES)
    categorical = list(categorical or config.CATEGORICAL_FEATURES)
    binary = list(binary or config.BINARY_FEATURES)

    numeric_pipe = Pipeline(steps=[("imputer", SimpleImputer(strategy="median"))])
    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "ordinal",
                OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
            ),
        ]
    )
    binary_pipe = Pipeline(steps=[("imputer", SimpleImputer(strategy="most_frequent"))])
    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, numeric),
            ("cat", categorical_pipe, categorical),
            ("bin", binary_pipe, binary),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def export_bins(binner: SupervisedBinner, path: Path) -> None:
    """Persist learned bin edges to JSON (auditable, language-agnostic)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(binner.to_dict(), fh, indent=2)


def get_lr_feature_names(preprocessor: ColumnTransformer) -> list[str]:
    """Return the post-transform LR feature names in column order."""
    return list(preprocessor.get_feature_names_out())
