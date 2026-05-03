# Loan credit risk — dataset

This repository contains only the **tabular data files** from Microsoft’s [r-server-loan-credit-risk](https://github.com/microsoft/r-server-loan-credit-risk) sample, retained under the **MIT License** (see `LICENSE`). The original solution code, ARM templates, and SQL/R scripts were removed so the data can be used in a separate project with a clean history.

## Files (`Data/`)

| File | Description |
|------|-------------|
| `Loan.txt` | Simulated loan-level records (development-scale sample). |
| `Borrower.txt` | Simulated borrower-level records (development-scale sample). |
| `Loan_Prod.txt` | Small production-style loan sample from the upstream template. |
| `Borrower_Prod.txt` | Small production-style borrower sample from the upstream template. |

Use `git clone` rather than “Download ZIP” if line endings matter for your tooling (per upstream guidance).

## Attribution

Data and original sample: Copyright (c) Microsoft Corporation. SPDX-License-Identifier: MIT. Full text in `LICENSE`.
