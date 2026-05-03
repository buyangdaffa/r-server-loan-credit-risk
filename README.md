# Loan credit risk

End-to-end credit-default risk pipeline on public synthetic data. Logistic
regression baseline plus calibrated LightGBM, leakage-safe supervised
binning, KS / lift / calibration metrics, SHAP explanations, and a
Streamlit app for single and batch scoring.

[![Reproduce](https://github.com/buyangdaffa/r-server-loan-credit-risk/actions/workflows/reproduce.yml/badge.svg)](https://github.com/buyangdaffa/r-server-loan-credit-risk/actions/workflows/reproduce.yml)

> Live demo: [buyangdaffa-r-server-loan-credit-risk.streamlit.app](https://buyangdaffa-r-server-loan-credit-risk.streamlit.app/)

## Highlights

- **Leakage-safe pipeline**: train/test split happens before any
  imputation, encoding, or supervised binning fit
  ([`src/features.py`](src/features.py), [`src/modeling.py`](src/modeling.py)).
- **Two models, one CLI**: `python -m src.pipeline run-all` fits a
  `LogisticRegression` baseline on binned features and a `LightGBMClassifier`
  on raw features, both wrapped with isotonic calibration.
- **Credit-flavored metrics**: AUC, PR-AUC, **KS**, Brier, decile lift, and
  reliability curves, plus quantile **score bands** with observed bad
  rates ([`src/evaluation.py`](src/evaluation.py),
  [`src/scoring.py`](src/scoring.py)).
- **Explainability**: SHAP `TreeExplainer` on the LightGBM booster
  ([`src/explain.py`](src/explain.py)).
- **Streamlit app**: form input, threshold slider, top SHAP drivers,
  and batch CSV scoring with a download button
  ([`app/streamlit_app.py`](app/streamlit_app.py)).
- **Reproducibility**: pinned [`requirements.txt`](requirements.txt),
  [`.python-version`](.python-version), optional
  [`environment.yml`](environment.yml) for Conda, and a CI workflow
  ([`.github/workflows/reproduce.yml`](.github/workflows/reproduce.yml))
  that runs the converter, full pytest suite, and a 1k-row training
  smoke test.

## Held-out metrics (full data, seed 42)

| Model | AUC | PR-AUC | KS | Brier |
|-------|----:|-------:|---:|------:|
| Logistic regression (binned) | 0.918 | 0.655 | 0.677 | 0.054 |
| LightGBM (calibrated) | 0.929 | 0.707 | 0.697 | 0.050 |

Full report regenerates to `outputs/reports/metrics.md`.

## Repo layout

```text
data/
  raw/         Tab-separated source files (tracked in git).
  processed/   UTF-8 CSVs from src/convert_data_to_csv.py.
  artifacts/   Trained models, bins, score bands, metadata (regenerable).
src/
  config.py    Paths, seed, feature schema, label.
  data.py      Load + merge + isBad derivation.
  features.py  Imputation, supervised binning, preprocessor builders.
  modeling.py  LR + LightGBM trainers, isotonic calibration, joblib I/O.
  evaluation.py  AUC, PR-AUC, KS, lift, Brier, calibration curve, report.
  explain.py   SHAP top-N drivers for the LightGBM booster.
  scoring.py   ScoreBand, score_one, score_batch.
  pipeline.py  CLI: run-all, train, evaluate, score.
  convert_data_to_csv.py   data/raw/*.txt -> data/processed/*.csv.
app/
  streamlit_app.py  Single application + batch scoring + model details.
tests/         pytest suite incl. end-to-end smoke test.
notebooks/     Exploratory notebooks; final logic lives in src/.
docs/MODEL_CARD.md   Data, label, metrics, limitations.
outputs/reports/     metrics.md / metrics.json (regenerable).
```

## Reproduce locally

1. **Python 3.11**, fresh virtual environment:

   ```bash
   python -m venv .venv
   source .venv/bin/activate     # Windows: .venv\Scripts\activate
   pip install -U pip
   pip install -r requirements.txt -r requirements-dev.txt
   ```

   Conda alternative: `conda env create -f environment.yml && conda activate loan-credit-risk`.

2. **Regenerate processed CSVs**:

   ```bash
   python src/convert_data_to_csv.py
   ```

3. **Train + evaluate** (writes models to `data/artifacts/`, report to `outputs/reports/`):

   ```bash
   python -m src.pipeline run-all
   ```

   For a fast smoke run: `python -m src.pipeline run-all --sample 5000 --skip-convert`.

4. **Run the Streamlit app**:

   ```bash
   streamlit run app/streamlit_app.py
   ```

5. **Tests**: `pytest`.

## Web app

Three tabs:

1. **Single application**: form prefilled with a real test-set record;
   pick a model and threshold, see probability, decision, score band, and
   top SHAP drivers.
2. **Batch scoring**: upload a CSV with the modeling-frame schema,
   download a scored CSV.
3. **Model details**: training metadata, held-out metrics, calibration
   curve, lift by decile.

## What I changed vs the upstream Microsoft sample

- Replaced the `RevoScaleR` + SQL Server R Services pipeline with a
  Python pipeline (`pandas`, `scikit-learn`, `lightgbm`, `shap`,
  `streamlit`) that runs on any laptop and on Streamlit Community Cloud.
- Reorganized the data into `data/raw/` and `data/processed/`, added a
  documented converter, pinned dependencies, and added `pyproject.toml`
  pytest config.
- Re-implemented the modeling steps as
  small, tested modules in `src/`, with leakage-safe supervised binning,
  isotonic calibration, KS-maximizing thresholding, score bands, SHAP
  explainability, and a model card under [`docs/`](docs/MODEL_CARD.md).
- Built a Streamlit web app for single and batch scoring on top of the
  saved artifacts, plus a CI workflow that runs the converter, the full
  pytest suite, and a 1k-row training smoke test.

## Attribution

Data and the original sample: Copyright (c) Microsoft Corporation,
SPDX-License-Identifier: MIT. Full license text in [`LICENSE`](LICENSE).
