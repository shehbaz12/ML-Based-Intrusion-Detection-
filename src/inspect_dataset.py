"""
inspect_dataset.py — Dataset inspection script.

Run this BEFORE any modeling to verify the actual column names, label values,
data types, and quality issues. Results are saved to:
  results/dataset_inspection_report.txt

Usage:
  .venv/Scripts/python.exe src/inspect_dataset.py
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src import config
from src.data_loader import load_raw_data, get_dataset_summary, validate_dataset, print_summary
import numpy as np


def run_inspection():
    output_lines = []

    def log(msg=""):
        print(msg)
        output_lines.append(str(msg))

    log("=" * 70)
    log("CIC-IDS2017 DATASET INSPECTION REPORT")
    log("=" * 70)

    # ── Load ─────────────────────────────────────────────────────────────────
    try:
        df = load_raw_data(config.RAW_DATA_DIR)
    except FileNotFoundError as e:
        log(f"ERROR: {e}")
        log("\nPlease download the dataset and place CSVs in data/raw/")
        log("See data/README.md for instructions.")
        sys.exit(1)

    # ── Column names ──────────────────────────────────────────────────────────
    log(f"\n{'─'*40}")
    log("COLUMN NAMES (exact, after whitespace strip):")
    log(f"{'─'*40}")
    for i, col in enumerate(df.columns):
        log(f"  {i:3d}: '{col}'")

    # ── Label column check ────────────────────────────────────────────────────
    log(f"\n{'─'*40}")
    log("LABEL COLUMN CHECK:")
    log(f"{'─'*40}")
    possible_label_cols = [c for c in df.columns if "label" in c.lower()
                           or "class" in c.lower()]
    log(f"  Possible label columns: {possible_label_cols}")

    if config.LABEL_COLUMN in df.columns:
        log(f"  config.LABEL_COLUMN = '{config.LABEL_COLUMN}' ✓ FOUND")
        log(f"\n  Unique values in '{config.LABEL_COLUMN}':")
        vc = df[config.LABEL_COLUMN].value_counts()
        for val, cnt in vc.items():
            log(f"    {val:<50} {cnt:>8,}  ({cnt/len(df)*100:.2f}%)")
    else:
        log(f"  config.LABEL_COLUMN = '{config.LABEL_COLUMN}' ✗ NOT FOUND")
        log(f"  >>> UPDATE config.py: LABEL_COLUMN = (one of {possible_label_cols})")

    # ── Summary ───────────────────────────────────────────────────────────────
    summary = get_dataset_summary(df)
    log(f"\n{'─'*40}")
    log("DATASET STATISTICS:")
    log(f"{'─'*40}")
    log(f"  Shape:         {summary['shape']}")
    log(f"  Memory:        {summary['memory_mb']} MB")
    log(f"  Total missing: {summary['total_missing']:,}")
    log(f"  Total inf:     {summary['total_inf']:,}")
    log(f"  Duplicates:    {summary['duplicate_rows']:,}")

    # ── Infinite values by column ─────────────────────────────────────────────
    if summary['inf_counts']:
        log(f"\n  Columns with infinite values:")
        for col, cnt in sorted(summary['inf_counts'].items(), key=lambda x: -x[1]):
            log(f"    {col:<50} {cnt:>8,}")

    # ── Missing values ────────────────────────────────────────────────────────
    missing = {k: v for k, v in summary['missing_counts'].items() if v > 0}
    if missing:
        log(f"\n  Columns with missing values:")
        for col, cnt in sorted(missing.items(), key=lambda x: -x[1]):
            log(f"    {col:<50} {cnt:>8,}")

    # ── Data types ────────────────────────────────────────────────────────────
    log(f"\n{'─'*40}")
    log("DATA TYPES:")
    log(f"{'─'*40}")
    dtypes = df.dtypes.value_counts()
    for dtype, cnt in dtypes.items():
        log(f"  {str(dtype):<15}: {cnt} columns")

    # ── Validation warnings ───────────────────────────────────────────────────
    log(f"\n{'─'*40}")
    log("VALIDATION WARNINGS:")
    log(f"{'─'*40}")
    warnings = validate_dataset(df)
    if warnings:
        for w in warnings:
            log(f"  ⚠ {w}")
    else:
        log("  ✓ No issues detected")

    # ── Identifier columns check ──────────────────────────────────────────────
    log(f"\n{'─'*40}")
    log("IDENTIFIER COLUMNS TO DROP (config.COLS_TO_DROP):")
    log(f"{'─'*40}")
    for col in config.COLS_TO_DROP:
        found = col in df.columns
        log(f"  {'✓' if found else '✗'} '{col}' {'(found)' if found else '(NOT FOUND — may need updating)'}")

    # ── Sample rows ───────────────────────────────────────────────────────────
    log(f"\n{'─'*40}")
    log("FIRST 3 ROWS (selected columns):")
    log(f"{'─'*40}")
    show_cols = list(df.columns[:5]) + ([config.LABEL_COLUMN]
                                         if config.LABEL_COLUMN in df.columns else [])
    log(df[show_cols].head(3).to_string())

    # ── Save report ───────────────────────────────────────────────────────────
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    report_path = config.RESULTS_DIR / "dataset_inspection_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(output_lines))

    log(f"\n{'='*70}")
    log(f"Report saved to: {report_path}")
    log(f"{'='*70}")

    return df


if __name__ == "__main__":
    run_inspection()
