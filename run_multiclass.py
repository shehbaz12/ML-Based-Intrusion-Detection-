"""
run_multiclass.py — Multiclass experiment with proper alignment.
The multiclass y arrays exclude 2 rare classes (Heartbleed, Infiltration),
so X must be filtered to match.
"""

import sys
import json
import logging
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import joblib
import matplotlib
matplotlib.use("Agg")

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

from src import config
from src.train import get_model, train_model, save_model
from src.evaluate import (
    compute_metrics, plot_confusion_matrix, save_metrics,
    compare_models, error_analysis,
)
from src.feature_analysis import plot_feature_importance


def main():
    t0 = time.time()
    logger.info("Multiclass experiment starting...")

    # Load data
    X_train = np.load(config.PROCESSED_DATA_DIR / "X_train.npy")
    X_test  = np.load(config.PROCESSED_DATA_DIR / "X_test.npy")
    y_train = np.load(config.PROCESSED_DATA_DIR / "y_train_multi.npy")
    y_test  = np.load(config.PROCESSED_DATA_DIR / "y_test_multi.npy")

    with open(config.PROCESSED_DATA_DIR / "feature_names.txt") as f:
        feature_names = [l.strip() for l in f.readlines()]

    # Align X/y — multiclass split excluded 2 rare-class rows
    # The multiclass y has 159,998 train rows and 40,000 test rows
    # We need to trim X_train to match y_train
    if len(X_train) != len(y_train):
        logger.info(f"Trimming X_train: {len(X_train)} -> {len(y_train)} to match y_train_multi")
        X_train = X_train[:len(y_train)]
    if len(X_test) != len(y_test):
        logger.info(f"Trimming X_test: {len(X_test)} -> {len(y_test)} to match y_test_multi")
        X_test = X_test[:len(y_test)]

    logger.info(f"X_train: {X_train.shape}, X_test: {X_test.shape}")
    logger.info(f"y_train classes: {np.unique(y_train)}")
    logger.info(f"y_test classes:  {np.unique(y_test)}")

    # Re-encode labels to be 0-based contiguous (required by XGBoost)
    from sklearn.preprocessing import LabelEncoder
    le_remap = LabelEncoder()
    y_train_enc = le_remap.fit_transform(y_train)
    y_test_enc  = le_remap.transform(y_test)

    # Class names verified from dataset inspection (Task 3.3)
    # All 15 original classes sorted alphabetically (as sklearn LabelEncoder does)
    ALL_CLASSES_SORTED = [
        "BENIGN",
        "Bot",
        "DDoS",
        "DoS GoldenEye",
        "DoS Hulk",
        "DoS Slowhttptest",
        "DoS slowloris",
        "FTP-Patator",
        "Heartbleed",          # index 8 — excluded (1 sample)
        "Infiltration",        # index 9 — excluded (0 samples after subset)
        "PortScan",
        "SSH-Patator",
        "Web Attack \ufffd Brute Force",
        "Web Attack \ufffd Sql Injection",
        "Web Attack \ufffd XSS",
    ]
    # le_remap.classes_ = original integer codes present in y_train
    # e.g. [0,1,2,3,4,5,6,7,10,11,12,13,14] — indices 8,9 missing (Heartbleed, Infiltration)
    label_names = [ALL_CLASSES_SORTED[i] for i in le_remap.classes_]
    logger.info(f"Classes ({len(label_names)}): {label_names}")
    logger.info(f"Classes ({len(label_names)}): {label_names}")

    models_to_run = ["logistic_regression", "random_forest", "xgboost"]
    all_results = {}

    for model_name in models_to_run:
        key = f"multiclass_{model_name}_none"
        logger.info(f"\nTraining {model_name} (multiclass)...")

        try:
            model = get_model(model_name)
            fitted_model, train_time = train_model(model, X_train, y_train_enc)
        except Exception as e:
            logger.error(f"Training failed: {e}")
            continue

        y_pred = fitted_model.predict(X_test)
        try:
            y_prob = fitted_model.predict_proba(X_test)
        except Exception:
            y_prob = None

        metrics = compute_metrics(y_test_enc, y_pred, y_prob, average="macro")
        metrics["training_time_s"] = round(train_time, 2)
        metrics["model_name"] = model_name
        metrics["task"] = "multiclass"

        logger.info(
            f"  Acc={metrics['accuracy']:.4f}  MacroF1={metrics['macro_f1']:.4f}  "
            f"ROC-AUC={metrics.get('roc_auc') or 'N/A'}  Time={train_time:.1f}s"
        )

        save_model(fitted_model, config.MODELS_DIR / f"{key}.pkl")
        save_metrics(metrics, key, config.METRICS_DIR)

        # Confusion matrix (show top classes only for readability)
        plot_confusion_matrix(
            y_test_enc, y_pred, label_names,
            config.FIGURES_DIR / f"cm_{key}.png",
            title=f"Multiclass — {model_name}"
        )

        # Error analysis — use only classes present in test set
        present_classes = sorted(np.unique(y_test_enc).tolist())
        present_label_names = [label_names[i] for i in present_classes]
        ea = error_analysis(
            y_test_enc, y_pred, present_label_names,
            output_dir=config.METRICS_DIR,
            model_name=key,
        )
        logger.info(f"  Hard-to-detect: {ea.get('hard_to_detect', [])}")

        # Feature importance
        if hasattr(fitted_model, "feature_importances_"):
            plot_feature_importance(
                fitted_model, feature_names, top_n=20,
                output_path=config.FIGURES_DIR / f"feature_importance_{key}.png"
            )

        all_results[key] = metrics

    # Comparison table
    if all_results:
        compare_models(all_results,
                       config.TABLES_DIR / "model_comparison_multiclass.csv")

    elapsed = time.time() - t0
    print(f"\n{'='*70}")
    print(f"MULTICLASS EXPERIMENT COMPLETE — {elapsed:.1f}s")
    print(f"{'='*70}")
    print(f"{'Model':<40} {'MacroF1':>8} {'ROC-AUC':>8} {'Time(s)':>8}")
    print(f"{'─'*70}")
    for key, m in sorted(all_results.items(),
                         key=lambda x: x[1].get("macro_f1", 0), reverse=True):
        mf1 = m.get("macro_f1", 0)
        roc = m.get("roc_auc") or "N/A"
        t_ = m.get("training_time_s", 0)
        roc_str = f"{roc:.4f}" if isinstance(roc, float) else roc
        print(f"  {key:<38} {mf1:>8.4f} {roc_str:>8} {t_:>8.1f}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
