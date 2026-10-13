"""
data_loader.py — Dataset ingestion for CIC-IDS2017.

Responsibilities:
- Discover and merge all CSV files from data/raw/
- Strip whitespace from column names (CIC-IDS2017 quirk)
- Cast numeric columns to float32 to reduce memory
- Report dataset statistics
- Raise FileNotFoundError if no CSVs found
"""

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def load_raw_data(data_dir: Path) -> pd.DataFrame:
    """
    Discover and merge all CSV files under data_dir.
    
    Postconditions:
    - Column names have leading/trailing whitespace stripped
    - Numeric columns cast to float32
    - Row count == sum of row counts across all individual CSVs
    - Label column present (as 'Label' after stripping)
    
    Raises:
        FileNotFoundError: if no .csv files exist under data_dir
    """
    csv_files = sorted(data_dir.glob("*.csv"))
    
    if not csv_files:
        raise FileNotFoundError(
            f"No CSV files found in {data_dir}.\n"
            f"Please download the CIC-IDS2017 dataset and place CSV files there.\n"
            f"See data/README.md for instructions."
        )
    
    logger.info(f"Found {len(csv_files)} CSV files in {data_dir}")
    
    dfs = []
    total_rows = 0
    for f in csv_files:
        logger.info(f"  Loading {f.name} ...")
        try:
            df_part = pd.read_csv(f, low_memory=False)
            # Strip whitespace from column names (CIC-IDS2017 quirk)
            df_part.columns = df_part.columns.str.strip()
            rows = len(df_part)
            total_rows += rows
            logger.info(f"    {rows:,} rows, {len(df_part.columns)} columns")
            dfs.append(df_part)
        except Exception as e:
            logger.warning(f"  Could not load {f.name}: {e}")
    
    if not dfs:
        raise FileNotFoundError(
            f"No CSV files could be loaded from {data_dir}. Check file permissions."
        )
    
    df = pd.concat(dfs, ignore_index=True)
    logger.info(f"Merged dataset: {len(df):,} rows x {len(df.columns)} columns")
    
    # Cast numeric columns to float32 to reduce memory footprint
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    df[numeric_cols] = df[numeric_cols].astype(np.float32)
    logger.info(f"Cast {len(numeric_cols)} numeric columns to float32")
    
    return df


def get_dataset_summary(df: pd.DataFrame) -> dict[str, Any]:
    """
    Return a comprehensive summary of the dataset.
    
    Returns dict with:
    - shape: (rows, cols)
    - columns: list of column names
    - dtypes: dict of col -> dtype
    - missing_values: dict of col -> count (only cols with missing)
    - label_distribution: value counts of Label column
    - memory_mb: memory usage in MB
    - infinite_counts: dict of col -> count of inf values
    """
    summary: dict[str, Any] = {}
    
    summary["shape"] = df.shape
    summary["columns"] = list(df.columns)
    summary["dtypes"] = {col: str(dtype) for col, dtype in df.dtypes.items()}
    
    # Missing values (only columns with at least 1 missing)
    missing = df.isnull().sum()
    summary["missing_values"] = missing[missing > 0].to_dict()
    summary["total_missing"] = int(missing.sum())
    
    # Infinite values in numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    inf_counts = {}
    for col in numeric_cols:
        n_inf = int(np.isinf(df[col]).sum())
        if n_inf > 0:
            inf_counts[col] = n_inf
    summary["infinite_counts"] = inf_counts
    summary["total_infinite"] = sum(inf_counts.values())
    
    # Duplicate rows
    summary["duplicate_rows"] = int(df.duplicated().sum())
    
    # Label distribution
    if "Label" in df.columns:
        vc = df["Label"].value_counts()
        summary["label_distribution"] = vc.to_dict()
        summary["label_proportions"] = (vc / len(df)).round(4).to_dict()
        summary["n_classes"] = len(vc)
    else:
        summary["label_distribution"] = {}
        summary["label_proportions"] = {}
        summary["n_classes"] = 0
    
    # Memory usage
    summary["memory_mb"] = round(df.memory_usage(deep=True).sum() / 1e6, 1)
    
    return summary


def validate_dataset(df: pd.DataFrame) -> list[str]:
    """
    Validate the dataset and return a list of warning strings.
    An empty list means no issues detected.
    """
    warnings = []
    
    if len(df) == 0:
        warnings.append("CRITICAL: DataFrame is empty.")
        return warnings
    
    if "Label" not in df.columns:
        warnings.append(
            "CRITICAL: 'Label' column not found. "
            f"Available columns: {list(df.columns[:10])}..."
        )
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) < 10:
        warnings.append(
            f"WARNING: Only {len(numeric_cols)} numeric columns found. "
            "Expected ~78 flow features."
        )
    
    if "Label" in df.columns:
        n_benign = (df["Label"] == "BENIGN").sum()
        if n_benign == 0:
            warnings.append("WARNING: No BENIGN samples found in Label column.")
        
        attack_types = df["Label"].unique()
        if len(attack_types) < 2:
            warnings.append(
                f"WARNING: Only {len(attack_types)} unique label(s) found. "
                "Expected 15 classes."
            )
    
    # Check for columns with all NaN
    all_nan = df.columns[df.isnull().all()].tolist()
    if all_nan:
        warnings.append(f"WARNING: Columns with all NaN values: {all_nan}")
    
    return warnings
