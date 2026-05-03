"""End-to-end CLI: regenerate processed data, train, evaluate, score.

Usage::

    python -m src.pipeline run-all                # full pipeline
    python -m src.pipeline run-all --sample 5000  # CI-friendly smoke run
    python -m src.pipeline train
    python -m src.pipeline evaluate
    python -m src.pipeline score path/to/inputs.csv --out path/to/scored.csv
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from . import config, data as data_mod, evaluation, features, modeling, scoring


# ---------------------------------------------------------------------------
# Pipeline steps
# ---------------------------------------------------------------------------


def step_convert() -> None:
    """Regenerate data/processed/*.csv from data/raw/*.txt."""
    subprocess.check_call(
        [sys.executable, str(config.ROOT / "src" / "convert_data_to_csv.py")]
    )


def step_train(*, sample: int | None = None) -> tuple[modeling.TrainSplit, dict]:
    """Train LR + LightGBM, calibrate, persist artifacts."""
    config.ensure_dirs()
    df = data_mod.build_modeling_frame()
    if sample is not None and sample < len(df):
        df = df.sample(n=sample, random_state=config.RANDOM_STATE).reset_index(drop=True)
    split = modeling.make_split(df)

    print(f"[train] rows: total={len(df):,} train={len(split.X_train):,} test={len(split.X_test):,}")
    print(f"[train] positive rate train={split.y_train.mean():.4f} test={split.y_test.mean():.4f}")

    print("[train] fitting logistic regression ...")
    lr_model = modeling.train_lr(split)
    modeling.save_model(lr_model, "lr_model")

    print("[train] fitting lightgbm ...")
    lgbm_model = modeling.train_lgbm(split)
    modeling.save_model(lgbm_model, "lgbm_model")

    binner = modeling.extract_lr_binner(lr_model)
    if binner is not None:
        features.export_bins(binner, config.DATA_ARTIFACTS / "bins.json")

    train_proba = {
        "lr": modeling.predict_proba(lr_model, split.X_train),
        "lgbm": modeling.predict_proba(lgbm_model, split.X_train),
    }
    bands = {
        name: scoring.fit_score_bands(split.y_train, p) for name, p in train_proba.items()
    }
    for name, band in bands.items():
        scoring.save_score_bands(band, config.DATA_ARTIFACTS / f"{name}_score_bands.json")

    modeling.write_metadata(
        split,
        extra={"trained_models": ["lr_model", "lgbm_model"], "sample": sample},
    )
    return split, {"lr": lr_model, "lgbm": lgbm_model}


def step_evaluate(split: modeling.TrainSplit, models: dict) -> dict:
    """Compute metrics for each model on the held-out split and write report."""
    metrics: dict[str, evaluation.Metrics] = {}
    for name, model in models.items():
        proba = modeling.predict_proba(model, split.X_test)
        metrics[name] = evaluation.evaluate(split.y_test, proba)
        m = metrics[name]
        print(
            f"[eval] {name:>4}: AUC={m.auc:.4f} PR-AUC={m.pr_auc:.4f} "
            f"KS={m.ks:.4f} Brier={m.brier:.4f} thr={m.threshold:.4f}"
        )

    json_path, md_path = evaluation.write_report(metrics, config.OUTPUTS_REPORTS)
    print(f"[eval] report: {md_path}")

    thresholds = {name: float(m.threshold) for name, m in metrics.items()}
    (config.DATA_ARTIFACTS / "thresholds.json").write_text(
        json.dumps(thresholds, indent=2), encoding="utf-8"
    )
    return {"metrics": metrics, "json": json_path, "md": md_path}


def step_score(input_path: Path, output_path: Path, *, model_name: str = "lgbm_model") -> Path:
    """Score a CSV with the saved model and write augmented CSV."""
    config.ensure_dirs()
    model = modeling.load_model(model_name)
    band_name = "lgbm" if model_name.startswith("lgbm") else "lr"
    band_path = config.DATA_ARTIFACTS / f"{band_name}_score_bands.json"
    band = scoring.load_score_bands(band_path) if band_path.is_file() else None
    threshold = _load_threshold(band_name)
    df = pd.read_csv(input_path)
    out = scoring.score_batch(model, df, threshold=threshold, band=band)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_path, index=False)
    print(f"[score] wrote {len(out):,} rows to {output_path}")
    return output_path


def _load_threshold(model_key: str, default: float = 0.5) -> float:
    path = config.DATA_ARTIFACTS / "thresholds.json"
    if not path.is_file():
        return default
    return float(json.loads(path.read_text(encoding="utf-8")).get(model_key, default))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p_all = sub.add_parser("run-all", help="convert -> train -> evaluate")
    p_all.add_argument("--sample", type=int, default=None, help="downsample rows (for CI)")
    p_all.add_argument("--skip-convert", action="store_true", help="reuse existing data/processed/*.csv")

    p_train = sub.add_parser("train", help="train LR + LightGBM")
    p_train.add_argument("--sample", type=int, default=None)

    sub.add_parser("evaluate", help="evaluate previously trained models on the standard split")

    p_score = sub.add_parser("score", help="score a CSV with the saved model")
    p_score.add_argument("input", type=Path)
    p_score.add_argument("--out", type=Path, required=True)
    p_score.add_argument("--model", default="lgbm_model", choices=["lr_model", "lgbm_model"])

    return parser


def main(argv: list[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)
    if args.command == "run-all":
        if not args.skip_convert:
            step_convert()
        split, models = step_train(sample=args.sample)
        step_evaluate(split, models)
    elif args.command == "train":
        step_train(sample=args.sample)
    elif args.command == "evaluate":
        df = data_mod.build_modeling_frame()
        split = modeling.make_split(df)
        models = {
            "lr": modeling.load_model("lr_model"),
            "lgbm": modeling.load_model("lgbm_model"),
        }
        step_evaluate(split, models)
    elif args.command == "score":
        step_score(args.input, args.out, model_name=args.model)
    else:
        raise ValueError(f"unknown command: {args.command}")


if __name__ == "__main__":
    main()
