"""
preprocessing.py — Data cleaning and leakage-free preprocessing pipeline.

Key design contract (leakage prevention):
  - fit_transform_pipeline() calls pipeline.fit_transform() on X_train ONLY
  - pipeline.transform() is called on X_test using the already-fitted parameters
  - apply_imbalance_strategy() is NEVER called with test data

Verified against actual CIC-IDS2017 ML CSV structure (Task 3.3):
  - No Flow ID / Source IP / Destination IP / Timestamp columns in ML CSV
  - Columns with leading spaces are stripped by data_loader.py
  - Infinite values present in Flow Bytes/s and Flow Packets/s
  - 256,684 duplicate rows across all files
"""

import logging
from pathlib import Path
from typing import Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder

from src import config

logger = logging.getLogger(__name__)


# ── Data Cleaning ─────────────────────────────────────────────────────────────

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the raw CIC-IDS2017 DataFrame.

    Steps (in order):
    1. Strip whitespace from column names (defensive — data_loader does this too)
    2. Replace +inf/-inf with NaN in all numeric columns
    3. Drop rows where Label is NaN
    4. Drop exact duplicate rows
    5. Drop identifier/irrelevant columns (config.COLS_TO_DROP)

    Returns cleaned DataFrame. Logs a before/after report.

    Postconditions:
    - No +inf/-inf values in numeric columns
    - No NaN in Label column
    - No exact duplicate rows
    - COLS_TO_DROP columns removed
    """
    df = df.copy()

    # Step 1: Normalize column names
    df.columns = df.columns.str.strip()

    n_rows_start = len(df)
    n_cols_start = len(df.columns)
    logger.info(f"clean_data() start: {n_rows_start:,} rows x {n_cols_start} cols")

    # Step 2: Replace infinite values
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    n_inf = 0
    for col in numeric_cols:
        mask = np.isinf(df[col])
        n_inf += int(mask.sum())
        df.loc[mask, col] = np.nan
    logger.info(f"  Replaced {n_inf:,} infinite values with NaN")

    # Step 3: Drop rows with missing Label
    n_before = len(df)
    df = df.dropna(subset=[config.LABEL_COLUMN])
    n_dropped_label = n_before - len(df)
    if n_dropped_label > 0:
        logger.warning(f"  Dropped {n_dropped_label:,} rows with missing Label")

    # Step 4: Drop duplicate rows
    n_before = len(df)
    df = df.drop_duplicates()
    n_dup = n_before - len(df)
    logger.info(f"  Dropped {n_dup:,} duplicate rows ({100.0*n_dup/n_before:.2f}%)")

    # Step 5: Drop identifier/irrelevant columns
    cols_dropped = []
    for col in config.COLS_TO_DROP:
        if col in df.columns:
            df = df.drop(columns=[col])
            cols_dropped.append(col)
    logger.info(f"  Dropped columns: {cols_dropped}")

    n_rows_end = len(df)
    n_cols_end = len(df.columns)
    pct_removed = 100.0 * (n_rows_start - n_rows_end) / n_rows_start

    logger.info(f"clean_data() end: {n_rows_end:,} rows x {n_cols_end} cols")
    logger.info(f"  Rows removed: {n_rows_start - n_rows_end:,} ({pct_removed:.2f}%)")

    # Print report to stdout as well (this function is called interactively)
    print(f"\n{'='*60}")
    print(f"DATA CLEANING REPORT")
    print(f"{'='*60}")
    print(f"  Rows before   : {n_rows_start:,}")
    print(f"  Rows after    : {n_rows_end:,}")
    print(f"  Rows removed  : {n_rows_start - n_rows_end:,} ({pct_removed:.2f}%)")
    print(f"  Cols before   : {n_cols_start}")
    print(f"  Cols after    : {n_cols_end}")
    print(f"  Inf→NaN subs  : {n_inf:,}")
    print(f"  Dupes removed : {n_dup:,}")
    print(f"  Cols dropped  : {cols_dropped}")
    print(f"{'='*60}\n")

    assert len(df) > 0, "clean_data(): DataFrame is empty after cleaning!"
    return df


# ── Target Engineering ────────────────────────────────────────────────────────

def engineer_targets(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add binary_label and multiclass_label columns.

    binary_label    : 0 = BENIGN, 1 = ATTACK
    multiclass_label: integer-encoded attack type (LabelEncoder fitted here,
                      saved to models/label_encoder.pkl for decoding later)

    Note: LabelEncoder is fitted on ALL labels here for inspection purposes.
    For proper leakage prevention in the experiment, engineer_targets() is called
    on the full cleaned dataset, then split_data() separates train/test.
    The LabelEncoder fitted here is saved and reused — not refitted on test labels.
    """
    df = df.copy()

    # Binary label
    df[config.BINARY_LABEL_COLUMN] = (
        df[config.LABEL_COLUMN] != config.BENIGN_CLASS
    ).astype(int)

    # Multiclass label
    le = LabelEncoder()
    df[config.MULTICLASS_LABEL_COLUMN] = le.fit_transform(df[config.LABEL_COLUMN])

    # Save LabelEncoder for decoding predictions later
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    le_path = config.MODELS_DIR / "label_encoder.pkl"
    joblib.dump(le, le_path)
    logger.info(f"LabelEncoder saved to {le_path}")

    # Print class distributions
    print(f"\nBINARY LABEL DISTRIBUTION:")
    vc_bin = df[config.BINARY_LABEL_COLUMN].value_counts().sort_index()
    total = len(df)
    for val, cnt in vc_bin.items():
        name = "BENIGN" if val == 0 else "ATTACK"
        print(f"  {val} ({name}): {cnt:,}  ({100.0*cnt/total:.2f}%)")

    print(f"\nMULTICLASS LABEL DISTRIBUTION ({len(le.classes_)} classes):")
    vc_multi = df[config.LABEL_COLUMN].value_counts()
    for label, cnt in vc_multi.items():
        encoded = le.transform([label])[0]
        print(f"  [{encoded:2d}] {label:<55s}: {cnt:,}  ({100.0*cnt/total:.3f}%)")

    return df, le


# ── Train/Test Split ──────────────────────────────────────────────────────────

def split_data(
    df: pd.DataFrame,
    target_column: str,
    test_size: float = config.TEST_SIZE,
    random_state: int = config.RANDOM_STATE,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Stratified train/test split. No preprocessing applied here — pure split only.

    Leakage prevention: splitting happens BEFORE any fit_transform_pipeline() call.

    Handles rare classes (e.g., Heartbleed with 11 samples) by falling back to
    non-stratified split if stratification fails.

    Returns: X_train, X_test, y_train, y_test
    """
    # Drop non-feature columns
    drop_cols = [
        config.LABEL_COLUMN,
        config.BINARY_LABEL_COLUMN,
        config.MULTICLASS_LABEL_COLUMN,
    ]
    feature_cols = [c for c in df.columns if c not in drop_cols]
    X = df[feature_cols]
    y = df[target_column]

    # Check for classes with < 2 samples (can't stratify)
    class_counts = y.value_counts()
    rare_classes = class_counts[class_counts < 2].index.tolist()
    if rare_classes:
        logger.warning(
            f"Classes with < 2 samples (cannot stratify): {rare_classes}. "
            f"Excluding from dataset."
        )
        mask = ~y.isin(rare_classes)
        X = X[mask]
        y = y[mask]
        print(f"  WARNING: Excluded {(~mask).sum()} rows with rare classes: {rare_classes}")

    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=test_size,
            random_state=random_state,
            stratify=y,
        )
    except ValueError as e:
        logger.warning(f"Stratified split failed ({e}), falling back to random split")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=test_size,
            random_state=random_state,
        )

    print(f"\nTRAIN/TEST SPLIT (target='{target_column}', test_size={test_size}):")
    print(f"  Train: {len(X_train):,} rows  |  Test: {len(X_test):,} rows")
    print(f"  Features: {X_train.shape[1]}")

    # Verify stratification
    train_dist = y_train.value_counts(normalize=True).sort_index()
    test_dist = y_test.value_counts(normalize=True).sort_index()
    print(f"  Train class proportions (top 3): "
          f"{dict(list(train_dist.items())[:3])}")
    print(f"  Test  class proportions (top 3): "
          f"{dict(list(test_dist.items())[:3])}")

    return X_train, X_test, y_train, y_test


# ── Preprocessing Pipeline ────────────────────────────────────────────────────

def build_preprocessing_pipeline(
    numeric_impute_strategy: str = "median",
    scaling: str = "standard",
) -> Pipeline:
    """
    Build an unfitted sklearn Pipeline:
      SimpleImputer(strategy=numeric_impute_strategy) → StandardScaler()

    Returns unfitted Pipeline. Call fit_transform_pipeline() to fit it.
    """
    if scaling == "standard":
        scaler = StandardScaler()
    elif scaling == "minmax":
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler()
    else:
        raise ValueError(f"Unknown scaling: {scaling}. Use 'standard' or 'minmax'.")

    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy=numeric_impute_strategy)),
        ("scaler", scaler),
    ])
    return pipeline


def fit_transform_pipeline(
    pipeline: Pipeline,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
) -> Tuple[np.ndarray, np.ndarray, Pipeline]:
    """
    Fit pipeline on X_train ONLY, then transform both X_train and X_test.

    LEAKAGE PREVENTION CONTRACT:
    - pipeline.fit_transform() is called on X_train rows only
    - The fitted scaler's mean/variance are derived exclusively from X_train
    - pipeline.transform() is then applied to X_test using those fitted parameters
    - pipeline.fit() is NEVER called on X_test

    Returns: X_train_scaled, X_test_scaled, fitted_pipeline
    """
    feature_names = list(X_train.columns)

    # Fit AND transform training data
    X_train_scaled = pipeline.fit_transform(X_train)

    # Transform test data using ALREADY-FITTED pipeline (no re-fitting)
    X_test_scaled = pipeline.transform(X_test)

    # Postcondition checks
    assert X_train_scaled.shape[1] == X_test_scaled.shape[1], (
        f"Feature dimension mismatch: train={X_train_scaled.shape[1]}, "
        f"test={X_test_scaled.shape[1]}"
    )
    assert X_train_scaled.shape[1] == len(feature_names), (
        f"Feature count mismatch: scaled={X_train_scaled.shape[1]}, "
        f"names={len(feature_names)}"
    )

    print(f"\nPREPROCESSING PIPELINE:")
    print(f"  Fitted on X_train ({X_train_scaled.shape[0]:,} rows)")
    print(f"  Transformed X_test ({X_test_scaled.shape[0]:,} rows)")
    print(f"  Feature dimensions: train={X_train_scaled.shape}, test={X_test_scaled.shape}")

    return X_train_scaled, X_test_scaled, pipeline


# ── Class Imbalance ───────────────────────────────────────────────────────────

def apply_imbalance_strategy(
    X_train: np.ndarray,
    y_train: np.ndarray,
    strategy: str,
    random_state: int = config.RANDOM_STATE,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply a class imbalance handling strategy to TRAINING DATA ONLY.

    Strategies:
    - "none"         : return unchanged
    - "class_weight" : handled at model level — return unchanged
    - "undersample"  : RandomUnderSampler (majority class reduced)
    - "smote"        : SMOTE synthesis (minority class augmented)

    CRITICAL: This function must NEVER be called with test data.
    SMOTE is applied AFTER split and AFTER scaling to prevent leakage.

    Returns: X_resampled, y_resampled
    """
    if strategy in ("none", "class_weight"):
        if strategy == "class_weight":
            logger.info("class_weight strategy: handled at model level, no resampling")
        return X_train, y_train

    _log_class_distribution("Before resampling", y_train)

    if strategy == "undersample":
        from imblearn.under_sampling import RandomUnderSampler
        sampler = RandomUnderSampler(random_state=random_state)
        X_res, y_res = sampler.fit_resample(X_train, y_train)

    elif strategy == "smote":
        from imblearn.over_sampling import SMOTE
        # SMOTE requires at least k_neighbors+1 samples per class
        # Check for classes too small for SMOTE (need ≥ 6 samples)
        unique, counts = np.unique(y_train, return_counts=True)
        min_count = counts.min()
        k_neighbors = min(5, min_count - 1)
        if k_neighbors < 1:
            logger.warning(
                f"SMOTE: some classes have too few samples (min={min_count}). "
                f"Falling back to k_neighbors=1."
            )
            k_neighbors = 1
        sampler = SMOTE(random_state=random_state, k_neighbors=k_neighbors)
        X_res, y_res = sampler.fit_resample(X_train, y_train)

    else:
        raise ValueError(
            f"Unknown imbalance strategy: '{strategy}'. "
            f"Choose from: {config.IMBALANCE_STRATEGIES}"
        )

    _log_class_distribution("After resampling", y_res)
    logger.info(
        f"apply_imbalance_strategy('{strategy}'): "
        f"{len(y_train):,} → {len(y_res):,} samples"
    )
    return X_res, y_res


def _log_class_distribution(label: str, y: np.ndarray) -> None:
    """Helper: print class distribution."""
    unique, counts = np.unique(y, return_counts=True)
    total = len(y)
    print(f"\n  {label} ({total:,} total):")
    for cls, cnt in zip(unique, counts):
        print(f"    Class {cls}: {cnt:,} ({100.0*cnt/total:.2f}%)")


# ── Subset Sampling ───────────────────────────────────────────────────────────

def sample_subset(
    df: pd.DataFrame,
    n: int = config.SUBSET_SIZE,
    target_column: str = config.BINARY_LABEL_COLUMN,
    random_state: int = config.RANDOM_STATE,
) -> pd.DataFrame:
    """
    Take a stratified subset of the dataset for memory-constrained machines.

    Documents the sampling for reproducibility:
    - Total rows available
    - Number sampled
    - Sampling seed used

    Returns the sampled DataFrame.
    """
    total = len(df)
    if total <= n:
        logger.info(f"Dataset ({total:,} rows) fits in memory — no sampling needed")
        return df

    # Stratified sample
    from sklearn.model_selection import train_test_split
    df_sample, _ = train_test_split(
        df,
        train_size=n,
        random_state=random_state,
        stratify=df[target_column],
    )

    print(f"\nSUBSET SAMPLING (memory constraint):")
    print(f"  Total rows available : {total:,}")
    print(f"  Rows sampled         : {len(df_sample):,}")
    print(f"  Sampling fraction    : {len(df_sample)/total:.3f}")
    print(f"  Random state         : {random_state}")
    print(f"  NOTE: All results apply to this subset, not the full dataset.")
    print(f"  To use full dataset, set SUBSET_SIZE > {total:,} in config.py")

    return df_sample.reset_index(drop=True)
