"""Performance metrics and reporting for credit-risk classifiers.

Metrics emphasized for credit risk:

- AUC (ROC) and PR-AUC for ranking quality.
- KS statistic for separation between good and bad applicants.
- Lift at deciles for marketing/operational interpretation.
- Brier score and reliability curve for probability calibration.
- Threshold tuning that maximizes KS, plus a confusion matrix at that cutoff.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)


@dataclass
class Metrics:
    auc: float
    pr_auc: float
    ks: float
    ks_threshold: float
    brier: float
    threshold: float
    confusion: dict = field(default_factory=dict)
    lift_by_decile: list[float] = field(default_factory=list)
    bad_rate_by_decile: list[float] = field(default_factory=list)
    calibration: dict = field(default_factory=dict)


def ks_statistic(y_true: np.ndarray, y_score: np.ndarray) -> tuple[float, float]:
    """Return (KS, threshold). KS is the max gap between TPR and FPR across cutoffs."""
    fpr, tpr, thr = roc_curve(y_true, y_score)
    diff = tpr - fpr
    idx = int(np.argmax(diff))
    return float(diff[idx]), float(thr[idx])


def decile_lift(y_true: np.ndarray, y_score: np.ndarray, *, n_bins: int = 10) -> tuple[list[float], list[float]]:
    """Lift and bad rate per score decile (highest score = decile 1)."""
    df = pd.DataFrame({"y": y_true, "p": y_score})
    df["decile"] = pd.qcut(-df["p"], q=n_bins, labels=False, duplicates="drop") + 1
    grouped = df.groupby("decile")["y"]
    base = df["y"].mean() if df["y"].mean() > 0 else 1e-9
    lifts = (grouped.mean() / base).reindex(range(1, n_bins + 1)).fillna(0.0).tolist()
    bad_rates = grouped.mean().reindex(range(1, n_bins + 1)).fillna(0.0).tolist()
    return [float(x) for x in lifts], [float(x) for x in bad_rates]


def evaluate(
    y_true,
    y_score,
    *,
    threshold: float | None = None,
    n_calibration_bins: int = 10,
) -> Metrics:
    y_true_arr = np.asarray(y_true).astype(int)
    y_score_arr = np.asarray(y_score).astype(float)

    auc = float(roc_auc_score(y_true_arr, y_score_arr))
    pr_auc = float(average_precision_score(y_true_arr, y_score_arr))
    ks_val, ks_thr = ks_statistic(y_true_arr, y_score_arr)
    brier = float(brier_score_loss(y_true_arr, y_score_arr))

    cutoff = float(threshold) if threshold is not None else ks_thr
    y_pred = (y_score_arr >= cutoff).astype(int)
    cm = confusion_matrix(y_true_arr, y_pred, labels=[0, 1])
    confusion = {
        "tn": int(cm[0, 0]),
        "fp": int(cm[0, 1]),
        "fn": int(cm[1, 0]),
        "tp": int(cm[1, 1]),
    }

    lift, bad_rate = decile_lift(y_true_arr, y_score_arr)

    prob_true, prob_pred = calibration_curve(
        y_true_arr,
        y_score_arr,
        n_bins=n_calibration_bins,
        strategy="quantile",
    )
    calibration = {
        "prob_pred": [float(x) for x in prob_pred],
        "prob_true": [float(x) for x in prob_true],
    }

    return Metrics(
        auc=auc,
        pr_auc=pr_auc,
        ks=ks_val,
        ks_threshold=ks_thr,
        brier=brier,
        threshold=cutoff,
        confusion=confusion,
        lift_by_decile=lift,
        bad_rate_by_decile=bad_rate,
        calibration=calibration,
    )


def metrics_to_dict(m: Metrics) -> dict:
    return asdict(m)


def write_report(
    metrics_by_model: dict[str, Metrics],
    out_dir: Path,
) -> tuple[Path, Path]:
    """Write a JSON dump and a Markdown summary to out_dir."""
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "metrics.json"
    md_path = out_dir / "metrics.md"

    payload = {name: metrics_to_dict(m) for name, m in metrics_by_model.items()}
    with json_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)

    lines: list[str] = ["# Loan credit risk - evaluation report", ""]
    lines.append("| Model | AUC | PR-AUC | KS | Brier | Threshold (KS-opt) |")
    lines.append("|-------|----:|-------:|---:|------:|------------------:|")
    for name, m in metrics_by_model.items():
        lines.append(
            f"| {name} | {m.auc:.4f} | {m.pr_auc:.4f} | {m.ks:.4f} | "
            f"{m.brier:.4f} | {m.threshold:.4f} |"
        )

    for name, m in metrics_by_model.items():
        lines.extend(["", f"## {name}", "", "### Confusion matrix at KS threshold", ""])
        lines.append("| | Pred 0 | Pred 1 |")
        lines.append("|---|---:|---:|")
        lines.append(f"| **Actual 0** | {m.confusion['tn']} | {m.confusion['fp']} |")
        lines.append(f"| **Actual 1** | {m.confusion['fn']} | {m.confusion['tp']} |")

        lines.extend(["", "### Lift by decile", "", "| Decile | Lift | Bad rate |", "|---:|---:|---:|"])
        for i, (lift, bad) in enumerate(zip(m.lift_by_decile, m.bad_rate_by_decile), start=1):
            lines.append(f"| {i} | {lift:.2f} | {bad:.4f} |")

    md_path.write_text("\n".join(lines), encoding="utf-8")
    return json_path, md_path
