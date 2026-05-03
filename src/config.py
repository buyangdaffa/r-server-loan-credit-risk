"""Project-wide configuration: paths, constants, feature schema."""

from __future__ import annotations

from pathlib import Path

# Repo root = parent of src/
ROOT = Path(__file__).resolve().parent.parent

DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
DATA_ARTIFACTS = ROOT / "data" / "artifacts"
OUTPUTS_REPORTS = ROOT / "outputs" / "reports"

# Reproducibility
RANDOM_STATE = 42

# Label
TARGET = "isBad"
LOAN_STATUS_COL = "loanStatus"
# loanStatus values that count as bad. Upstream sample defines bad loans as
# charged-off or defaulted; in this dataset the bad bucket is "Default".
BAD_STATUS_VALUES: tuple[str, ...] = ("Charged Off", "Default")

# Identifiers (dropped before modeling).
ID_COLS: tuple[str, ...] = ("loanId", "memberId")

# Date column dropped for v1 (could be feature-engineered later).
DROP_COLS: tuple[str, ...] = ID_COLS + (LOAN_STATUS_COL, "date")

# Numeric features used for supervised binning + LR baseline.
NUMERIC_FEATURES: tuple[str, ...] = (
    "loanAmount",
    "interestRate",
    "monthlyPayment",
    "annualIncome",
    "dtiRatio",
    "lengthCreditHistory",
    "numTotalCreditLines",
    "numOpenCreditLines",
    "numOpenCreditLines1Year",
    "revolvingBalance",
    "revolvingUtilizationRate",
    "numDerogatoryRec",
    "numDelinquency2Years",
    "numChargeoff1year",
    "numInquiries6Mon",
)

# Low-cardinality categorical features (one-hot encoded).
CATEGORICAL_FEATURES: tuple[str, ...] = (
    "purpose",
    "term",
    "grade",
    "residentialState",
    "yearsEmployment",
    "homeOwnership",
)

# Binary 0/1 columns kept as-is.
BINARY_FEATURES: tuple[str, ...] = (
    "isJointApplication",
    "incomeVerified",
)

ALL_FEATURES: tuple[str, ...] = (
    NUMERIC_FEATURES + CATEGORICAL_FEATURES + BINARY_FEATURES
)

# Default split.
TEST_SIZE = 0.25

# Score band quantiles (percentile cutoffs reported alongside predictions).
SCORE_BAND_PERCENTILES: tuple[float, ...] = tuple(round(0.05 * i, 2) for i in range(1, 20))


def ensure_dirs() -> None:
    """Make sure write directories exist."""
    for path in (DATA_PROCESSED, DATA_ARTIFACTS, OUTPUTS_REPORTS):
        path.mkdir(parents=True, exist_ok=True)
