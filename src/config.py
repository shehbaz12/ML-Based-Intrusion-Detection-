"""
Central configuration for the ML Intrusion Detection experiment.
All paths, hyperparameters, and constants are defined here.
RANDOM_STATE = 42 is used everywhere for reproducibility.
"""

from pathlib import Path

# ── Reproducibility ──────────────────────────────────────────────────────────
RANDOM_STATE: int = 42
TEST_SIZE: float = 0.2
CV_FOLDS: int = 5
SUBSET_SIZE: int = 200_000  # rows to sample on memory-constrained machines (~8 GB RAM)

# ── Project root (one level up from this file) ───────────────────────────────
def get_project_root() -> Path:
    return Path(__file__).parent.parent

PROJECT_ROOT = get_project_root()

# ── Directory paths ───────────────────────────────────────────────────────────
DATA_DIR          = PROJECT_ROOT / "data"
RAW_DATA_DIR      = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR        = PROJECT_ROOT / "models"
RESULTS_DIR       = PROJECT_ROOT / "results"
FIGURES_DIR       = PROJECT_ROOT / "results" / "figures"
TABLES_DIR        = PROJECT_ROOT / "results" / "tables"
METRICS_DIR       = PROJECT_ROOT / "results" / "metrics"
REPORT_DIR        = PROJECT_ROOT / "report"

# ── Dataset column names (CIC-IDS2017 quirks) ────────────────────────────────
# VERIFIED by inspection (Task 3.3):
# - Raw column names have leading spaces (e.g., ' Destination Port')
# - data_loader.py strips all whitespace → clean names like 'Destination Port'
# - Label column is at index 78, named 'Label' after stripping
# - BENIGN class is spelled exactly 'BENIGN' (all caps)
# - Web Attack classes use Unicode replacement char: 'Web Attack \ufffd Brute Force'
LABEL_COLUMN: str = "Label"          # after whitespace stripping (verified)
BENIGN_CLASS: str = "BENIGN"         # verified from label distribution
BINARY_LABEL_COLUMN: str = "binary_label"
MULTICLASS_LABEL_COLUMN: str = "multiclass_label"

# Identifier columns to drop before modeling (not predictive features)
# VERIFIED against actual dataset columns during inspection (Task 3.3):
# The MachineLearningCSV version of CIC-IDS2017 does NOT contain Flow ID,
# Source IP, Destination IP, or Timestamp — those exist only in PCAP-derived files.
# The ML CSV starts directly at flow-level numeric features.
# We drop Destination Port (high cardinality, can leak port-specific patterns).
COLS_TO_DROP: list = [
    "Destination Port",  # high cardinality — verified present in actual data
    "Fwd Header Length.1",  # duplicate of Fwd Header Length (column 55 == column 34)
]

# ── Model configurations ──────────────────────────────────────────────────────
# All hyperparameters documented here for reproducibility.
MODEL_CONFIGS: dict = {
    "logistic_regression": {
        "max_iter": 1000,
        "random_state": RANDOM_STATE,
        "n_jobs": -1,
    },
    "random_forest": {
        "n_estimators": 100,
        "random_state": RANDOM_STATE,
        "n_jobs": -1,
        "max_depth": None,
    },
    "hist_gradient_boosting": {
        "random_state": RANDOM_STATE,
        "max_iter": 100,
    },
    "xgboost": {
        "tree_method": "hist",   # CPU-optimized
        "random_state": RANDOM_STATE,
        "n_estimators": 100,
        "n_jobs": -1,
        "eval_metric": "logloss",
        "verbosity": 0,
    },
}

# ── Imbalance strategies ──────────────────────────────────────────────────────
IMBALANCE_STRATEGIES: list = ["none", "class_weight", "undersample", "smote"]

# ── Figure settings ───────────────────────────────────────────────────────────
FIGURE_DPI: int = 300
FIGURE_FORMAT: str = "png"
