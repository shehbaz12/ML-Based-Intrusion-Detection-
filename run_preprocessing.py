"""
run_preprocessing.py — Phase 4: Data cleaning, target engineering, splitting,
and pipeline fitting. Processes data in batches to handle memory constraints.

Outputs:
  data/processed/X_train.npy, X_test.npy, y_train_binary.npy, y_test_binary.npy
  data/processed/y_train_multi.npy, y_test_multi.npy
  data/processed/feature_names.txt
  models/preprocessing_pipeline.pkl
  models/label_encoder.pkl
"""

import sys
import gc
import logging
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd
import joblib

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)

from src import config
from src.preprocessing import (
    clean_data, engineer_targets, sample_subset,
    split_data, build_preprocessing_pipeline, fit_transform_pipeline,
)

def main():
    t0 = time.time()
    config.PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # ── Load and clean one file at a time ─────────────────────────────────
    print("\n" + "="*60)
    print("PHASE 4: DATA PREPROCESSING (batch mode)")
    print("="*60)

    csv_files = sorted(config.RAW_DATA_DIR.glob("*.csv"))
    print(f"Processing {len(csv_files)} CSV files one at a time...\n")

    all_frames = []
    total_raw = 0
    total_clean = 0

    for f in csv_files:
        print(f"\n>> Loading {f.name}")
        df_raw = pd.read_csv(f, low_memory=False)
        df_raw.columns = df_raw.columns.str.strip()
        numeric_cols = df_raw.select_dtypes(include=[np.number]).columns
        df_raw[numeric_cols] = df_raw[numeric_cols].astype(np.float32)
        total_raw += len(df_raw)
        print(f"   Raw rows: {len(df_raw):,}")

        df_clean = clean_data(df_raw)
        total_clean += len(df_clean)
        all_frames.append(df_clean)

        del df_raw
        gc.collect()

    print(f"\n{'='*60}")
    print(f"COMBINED CLEANING SUMMARY")
    print(f"  Total raw rows  : {total_raw:,}")
    print(f"  Total clean rows: {total_clean:,}")
    print(f"  Total removed   : {total_raw - total_clean:,} ({100*(total_raw-total_clean)/total_raw:.2f}%)")
    print(f"{'='*60}")

    # ── Combine all cleaned files ──────────────────────────────────────────
    print("\nCombining all cleaned files...")
    df = pd.concat(all_frames, ignore_index=True)
    del all_frames
    gc.collect()
    print(f"Combined: {len(df):,} rows x {len(df.columns)} columns")

    # ── Engineer targets ───────────────────────────────────────────────────
    print("\nEngineering targets...")
    df, le = engineer_targets(df)

    # ── Subset sampling (memory constraint on ~8GB RAM) ────────────────────
    print(f"\nSubset sampling to {config.SUBSET_SIZE:,} rows for memory constraint...")
    df = sample_subset(df, n=config.SUBSET_SIZE,
                       target_column=config.BINARY_LABEL_COLUMN)
    gc.collect()

    # ── Binary classification split ────────────────────────────────────────
    print("\n" + "="*60)
    print("BINARY CLASSIFICATION SPLIT")
    print("="*60)
    X_train, X_test, y_train_bin, y_test_bin = split_data(
        df,
        target_column=config.BINARY_LABEL_COLUMN,
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_STATE,
    )

    # ── Multiclass split (same features, different labels) ─────────────────
    print("\n" + "="*60)
    print("MULTICLASS CLASSIFICATION SPLIT")
    print("="*60)
    _, _, y_train_multi, y_test_multi = split_data(
        df,
        target_column=config.MULTICLASS_LABEL_COLUMN,
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_STATE,
    )

    feature_names = list(X_train.columns)
    print(f"\nFeature count: {len(feature_names)}")

    # ── Fit preprocessing pipeline (on training data ONLY) ─────────────────
    print("\n" + "="*60)
    print("FITTING PREPROCESSING PIPELINE (train data only)")
    print("="*60)
    pipeline = build_preprocessing_pipeline()
    X_train_scaled, X_test_scaled, fitted_pipeline = fit_transform_pipeline(
        pipeline, X_train, X_test
    )

    # Verify no leakage
    assert X_train_scaled.shape[1] == X_test_scaled.shape[1]
    assert X_train_scaled.shape[1] == len(feature_names)
    print("  Leakage check: PASSED (pipeline fitted on train only)")

    # ── Save all artifacts ─────────────────────────────────────────────────
    print("\nSaving processed data to data/processed/ ...")
    np.save(config.PROCESSED_DATA_DIR / "X_train.npy", X_train_scaled)
    np.save(config.PROCESSED_DATA_DIR / "X_test.npy", X_test_scaled)
    np.save(config.PROCESSED_DATA_DIR / "y_train_binary.npy", y_train_bin.values)
    np.save(config.PROCESSED_DATA_DIR / "y_test_binary.npy", y_test_bin.values)
    np.save(config.PROCESSED_DATA_DIR / "y_train_multi.npy", y_train_multi.values)
    np.save(config.PROCESSED_DATA_DIR / "y_test_multi.npy", y_test_multi.values)

    # Save feature names
    feat_path = config.PROCESSED_DATA_DIR / "feature_names.txt"
    with open(feat_path, "w") as f:
        f.write("\n".join(feature_names))
    print(f"  Feature names saved to {feat_path}")

    # Save pipeline
    pipeline_path = config.MODELS_DIR / "preprocessing_pipeline.pkl"
    joblib.dump(fitted_pipeline, pipeline_path)
    print(f"  Pipeline saved to {pipeline_path}")

    elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print(f"PREPROCESSING COMPLETE in {elapsed:.1f}s")
    print(f"  X_train: {X_train_scaled.shape}")
    print(f"  X_test : {X_test_scaled.shape}")
    print(f"  y_train_binary distribution: "
          f"{dict(zip(*np.unique(y_train_bin, return_counts=True)))}")
    print(f"  y_test_binary distribution : "
          f"{dict(zip(*np.unique(y_test_bin, return_counts=True)))}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
