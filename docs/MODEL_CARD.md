# Model card: Loan credit risk

## Intended use

Demonstration model that estimates the probability of a loan defaulting,
given borrower- and loan-level features. Built as a portfolio project from
public synthetic data; **not intended for real lending decisions**.

## Models

Two calibrated classifiers are trained on the same train/test split and
saved to `data/artifacts/`:

| Name | Family | Preprocessing | Calibration |
|------|--------|---------------|-------------|
| `lr_model.joblib` | Logistic regression | Median imputation, supervised binning (DecisionTreeClassifier per numeric column), one-hot encoding | Isotonic (3-fold) |
| `lgbm_model.joblib` | LightGBM | Median imputation, ordinal-encoded categoricals | Isotonic (3-fold) |

LR is kept as the **interpretable baseline** in the spirit of credit-scoring
practice; LightGBM is the **performance model**.

## Data

Public synthetic data from Microsoft's
[`r-server-loan-credit-risk`](https://github.com/microsoft/r-server-loan-credit-risk)
sample, redistributed under MIT (see `LICENSE`).

| File | Rows | Role |
|------|-----:|------|
| `data/raw/Loan.txt` + `Borrower.txt` | 100,000 | Modeling (development sample) |
| `data/raw/Loan_Prod.txt` + `Borrower_Prod.txt` | ~22 | Small "production" demo set |

Schema is captured in [`src/config.py`](../src/config.py): 15 numeric
features, 6 categorical features, 2 binary features. Tab-separated raw
files are converted to UTF-8 CSVs by
[`src/convert_data_to_csv.py`](../src/convert_data_to_csv.py).

## Label

Binary `isBad`, derived from `loanStatus`:

- `1` if `loanStatus in {"Charged Off", "Default"}`
- `0` if `loanStatus == "Current"`
- `NaN` (row dropped) if missing or empty

In the supplied dataset, ~10% of loans are `Default`, none are
`Charged Off`. Positive class is the **bad** loan.

## Train/test split

- Stratified split, `test_size = 0.25`, `random_state = 42`.
- 75,000 training rows / 25,000 test rows.
- Same positive rate (~10.0%) on both sides.

Bins, imputers, and one-hot encoders are fit **only on the training fold**;
the test fold is transformed with the trained pipeline. No information from
the test fold leaks into preprocessing or the supervised binning.

## Held-out metrics (full data, seed 42)

| Model | AUC | PR-AUC | KS | Brier | KS-optimal threshold |
|-------|----:|-------:|---:|------:|---------------------:|
| LR | 0.9178 | 0.6552 | 0.6765 | 0.0540 | 0.1060 |
| LightGBM | 0.9292 | 0.7074 | 0.6968 | 0.0496 | 0.0955 |

Full report (confusion matrix at KS threshold, lift by decile, calibration)
is regenerated to `outputs/reports/metrics.md` and `metrics.json` by
`python -m src.pipeline run-all`.

## Operational thresholds and score bands

`fit_score_bands` quantizes the training-set probabilities into 19 quantile
cutoffs (5%-95% in 5% steps) and records the observed bad rate above each
cutoff. The Streamlit app surfaces the band index and bad rate alongside
each prediction so reviewers can read the score in business terms.

The default threshold persisted in `data/artifacts/thresholds.json` is the
**KS-maximizing** cutoff per model, but the app exposes a slider so users
can pick a stricter or looser policy.

## Explainability

SHAP `TreeExplainer` is run on the LightGBM booster (the calibrator above
it is monotone, so driver ranking is preserved). The Streamlit app shows
the top-N signed drivers for each prediction in [`src/explain.py`](../src/explain.py).

## Limitations and caveats

- **Synthetic data**: distributions may not match any real lender; numeric
  generality is illustrative.
- **Label is `Default`-only**: the upstream sample also defines `Charged Off`
  as bad, but the data we ship does not contain that value.
- **No protected-class fairness analysis**: the dataset includes
  `residentialState` only; no race, gender, or age. Do not deploy as-is.
- **Threshold pick is KS-maximizing**: it does not encode any cost matrix;
  real lenders should choose thresholds based on expected loss and
  regulatory constraints.
- **Calibration is global**: subgroup calibration was not validated.
- **Time stability**: there is no temporal split; concept drift is not
  modeled. The `date` column is dropped in v1.

## Reproducibility

- Code: this repository, version pinned in
  [`requirements.txt`](../requirements.txt) and
  [`requirements-dev.txt`](../requirements-dev.txt); Python 3.11 in CI.
- Seed: `RANDOM_STATE = 42` ([`src/config.py`](../src/config.py)).
- Run metadata (timestamp, git SHA, library versions, row counts) is
  recorded to `data/artifacts/metadata.json` at training time.
- CI ([`.github/workflows/reproduce.yml`](../.github/workflows/reproduce.yml))
  runs the converter, full test suite, and a 1k-row training smoke test on
  every push.
