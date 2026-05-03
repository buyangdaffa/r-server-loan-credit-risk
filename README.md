# Loan credit risk — dataset

This repository contains only the **tabular data files** from Microsoft’s [r-server-loan-credit-risk](https://github.com/microsoft/r-server-loan-credit-risk) sample, retained under the **MIT License** (see `LICENSE`). The original solution code, ARM templates, and SQL/R scripts were removed so the data can be used in a separate project with a clean history.

## Layout

| Path | Contents |
|------|----------|
| `data/raw/` | Tab-separated `.txt` from the upstream sample (source of truth in git). |
| `data/processed/` | UTF-8 `.csv` generated from raw files (e.g. `python src/convert_data_to_csv.py`). |

## Raw files (`data/raw/`)

| File | Description |
|------|-------------|
| `Loan.txt` | Simulated loan-level records (development-scale sample). |
| `Borrower.txt` | Simulated borrower-level records (development-scale sample). |
| `Loan_Prod.txt` | Small production-style loan sample from the upstream template. |
| `Borrower_Prod.txt` | Small production-style borrower sample from the upstream template. |

Use `git clone` rather than “Download ZIP” if line endings matter for your tooling (per upstream guidance).

## Reproduce the environment

1. **Python** — Use **3.10+** (CI uses **3.11**; [`.python-version`](.python-version) pins **3.11.9** for [pyenv](https://github.com/pyenv/pyenv) / [uv](https://docs.astral.sh/uv/) users).
2. **Virtual environment** (recommended):
   ```bash
   python -m venv .venv
   source .venv/bin/activate   # Windows: .venv\Scripts\activate
   pip install -U pip
   pip install -r requirements.txt
   ```
3. **Optional (notebooks + plots)** — `pip install -r requirements-dev.txt`
4. **Regenerate processed data** — `python src/convert_data_to_csv.py` writes CSVs under `data/processed/`.

Pinned versions live in [`requirements.txt`](requirements.txt) so installs stay consistent; bump pins intentionally when you upgrade. CI runs the converter and checks outputs on every push (see [`.github/workflows/reproduce.yml`](.github/workflows/reproduce.yml)).

## Attribution

Data and original sample: Copyright (c) Microsoft Corporation. SPDX-License-Identifier: MIT. Full text in `LICENSE`.
