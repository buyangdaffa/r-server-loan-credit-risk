"""Convert tab-separated raw .txt files under data/raw/ to CSV in data/processed/."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

# Repo root = parent of src/
ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RAW_DIR = ROOT / "data" / "raw"
DEFAULT_PROCESSED_DIR = ROOT / "data" / "processed"

# Basenames only; paths resolved under DEFAULT_RAW_DIR
DEFAULT_TXT_FILES = (
    "Loan.txt",
    "Borrower.txt",
    "Loan_Prod.txt",
    "Borrower_Prod.txt",
)


def txt_to_csv_path(txt_path: Path, out_dir: Path) -> Path:
    return out_dir / (txt_path.stem + ".csv")


def convert_txt_to_csv(
    txt_path: Path,
    out_path: Path,
    *,
    sep: str = "\t",
    encoding: str = "utf-8",
) -> pd.DataFrame:
    """Read a tab-separated .txt file and write UTF-8 CSV."""
    df = pd.read_csv(txt_path, sep=sep, encoding=encoding)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False, encoding="utf-8")
    return df


def convert_all(
    raw_dir: Path = DEFAULT_RAW_DIR,
    processed_dir: Path = DEFAULT_PROCESSED_DIR,
) -> list[Path]:
    written: list[Path] = []
    for name in DEFAULT_TXT_FILES:
        src = raw_dir / name
        if not src.is_file():
            raise FileNotFoundError(f"Missing input file: {src}")
        dst = txt_to_csv_path(src, processed_dir)
        convert_txt_to_csv(src, dst)
        written.append(dst)
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=DEFAULT_RAW_DIR,
        help="Directory with tab-separated .txt files (default: repo data/raw/)",
    )
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=DEFAULT_PROCESSED_DIR,
        help="Output directory for .csv files (default: repo data/processed/)",
    )
    parser.add_argument(
        "files",
        nargs="*",
        type=Path,
        help="Optional specific .txt paths; default converts all known sample files",
    )
    args = parser.parse_args()

    raw_dir = args.raw_dir.resolve()
    processed_dir = args.processed_dir.resolve()

    if args.files:
        for raw in args.files:
            p = Path(raw)
            src = next(
                (c for c in (p.resolve(), raw_dir / p.name) if c.is_file()),
                None,
            )
            if src is None:
                raise FileNotFoundError(f"Not found: {raw}")
            dst = txt_to_csv_path(src, processed_dir)
            convert_txt_to_csv(src, dst)
            print(dst)
    else:
        for name in DEFAULT_TXT_FILES:
            src = raw_dir / name
            dst = txt_to_csv_path(src, processed_dir)
            convert_txt_to_csv(src, dst)
            print(dst)


if __name__ == "__main__":
    main()
