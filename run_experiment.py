"""
run_experiment.py — Main experiment entry point.

Runs the complete ML pipeline:
  Load processed data → train models → evaluate → feature analysis →
  error analysis → model comparison → research findings

Usage:
  .venv/Scripts/python.exe run_experiment.py --task binary
  .venv/Scripts/python.exe run_experiment.py --task multiclass

All results saved to results/metrics/, results/figures/, results/tables/
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

from src import config
from src.train import get_model, train_model, save_model
from src.evaluate import (
    compute_metrics, plot_confusion_matrix, plot_roc_curve,
    plot_pr_curve, save_metrics, compare_models, error_analysis,
)
from src.feature_analysis import (
    plot_feature_importance, select_features_by_importance,
)
from src.preprocessing import apply_imbalance_strategy


def load_data(task: str):
    """Load preprocessed arrays from data/processed/."""
    X_train = np.load(config.PROCESSED_DATA_DIR / "X_train.npy")
    X_test  = np.load(config.PROCESSED_DATA_DIR / "X_test.npy")

    if task == "binary":
        y_train = np.load(config.PROCESSED_DATA_DIR / "y_train_binary.npy")
        y_test  = np.load(config.PROCESSED_DATA_DIR / "y_test_binary.npy")
    else:
        y_train = np.load(config.PROCESSED_DATA_DIR / "y_train_multi.npy")
        y_test  = np.load(config.PROCESSED_DATA_DIR / "y_test_multi.npy")

    with open(config.PROCESSED_DATA_DIR / "feature_names.txt") as f:
        feature_names = [l.strip() for l in f.readlines()]

    logger.info(f"Loaded: X_train={X_train.shape}, X_test={X_test.shape}, task={task}")
    return X_train, X_test, y_train, y_test, feature_names


def run_single_experiment(
    model_name: str,
    strategy: str,
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    feature_names: list,
    task: str,
    label_names: list,
) -> dict:
    """
    Train and evaluate one (model, strategy) combination.
    Returns metrics dict. Saves model, metrics, and figures.
    """
    key = f"{model_name}_{strategy}"
    logger.info(f"\n{'─'*50}")
    logger.info(f"Experiment: {key}")

    # Apply imbalance strategy (training data only)
    try:
        X_tr, y_tr = apply_imbalance_strategy(X_train, y_train, strategy)
    except Exception as e:
        logger.error(f"Imbalance strategy '{strategy}' failed: {e}")
        return {}

    # class_weight handled at model level
    cw = "balanced" if strategy == "class_weight" else None

    # Skip class_weight for models that don't support it
    if strategy == "class_weight" and model_name == "hist_gradient_boosting":
        logger.info(f"Skipping class_weight for HistGradientBoosting")
        return {}

    # Instantiate model
    try:
        model = get_model(model_name, class_weight=cw)
    except Exception as e:
        logger.error(f"Could not create model '{model_name}': {e}")
        return {}

    # Train
    try:
        fitted_model, train_time = train_model(model, X_tr, y_tr)
    except Exception as e:
        logger.error(f"Training failed for {key}: {e}")
        return {}

    # Predict
    y_pred = fitted_model.predict(X_test)
    try:
        y_prob = fitted_model.predict_proba(X_test)
    except Exception:
        y_prob = None

    # Metrics
    average = "binary" if task == "binary" else "macro"
    metrics = compute_metrics(y_test, y_pred, y_prob, average=average)
    metrics["training_time_s"] = round(train_time, 2)
    metrics["model_name"] = model_name
    metrics["strategy"] = strategy
    metrics["task"] = task

    logger.info(
        f"  Acc={metrics['accuracy']:.4f}  F1={metrics['f1']:.4f}  "
        f"MacroF1={metrics['macro_f1']:.4f}  "
        f"ROC-AUC={metrics.get('roc_auc') or 'N/A'}  "
        f"Time={train_time:.1f}s"
    )

    # Save model
    save_model(fitted_model, config.MODELS_DIR / f"{key}.pkl")

    # Save metrics
    save_metrics(metrics, key, config.METRICS_DIR)

    # Confusion matrix
    cm_path = config.FIGURES_DIR / f"cm_{key}.png"
    plot_confusion_matrix(y_test, y_pred, label_names, cm_path,
                          title=f"{model_name} ({strategy})")

    # ROC / PR curves (binary only with probabilities)
    if task == "binary" and y_prob is not None:
        plot_roc_curve(y_test, y_prob, key,
                       config.FIGURES_DIR / f"roc_{key}.png")
        plot_pr_curve(y_test, y_prob, key,
                      config.FIGURES_DIR / f"pr_{key}.png")

    return {key: metrics}


def main():
    parser = argparse.ArgumentParser(description="ML Intrusion Detection Experiment")
    parser.add_argument("--task", choices=["binary", "multiclass"],
                        default="binary", help="Classification task")
    parser.add_argument("--models", nargs="+",
                        default=list(config.MODEL_CONFIGS.keys()),
                        help="Models to run")
    parser.add_argument("--strategies", nargs="+",
                        default=config.IMBALANCE_STRATEGIES,
                        help="Imbalance strategies to run")
    args = parser.parse_args()

    t_total = time.time()
    logger.info(f"Starting experiment: task={args.task}")

    # Create output dirs
    for d in [config.METRICS_DIR, config.FIGURES_DIR, config.TABLES_DIR,
              config.MODELS_DIR]:
        d.mkdir(parents=True, exist_ok=True)

    # Load data
    X_train, X_test, y_train, y_test, feature_names = load_data(args.task)

    # Label names for confusion matrix
    le_path = config.MODELS_DIR / "label_encoder.pkl"
    if args.task == "multiclass" and le_path.exists():
        le = joblib.load(le_path)
        label_names = list(le.classes_)
    else:
        label_names = ["BENIGN", "ATTACK"]

    # ── Run all experiments ────────────────────────────────────────────────
    all_results = {}

    for strategy in args.strategies:
        for model_name in args.models:
            try:
                result = run_single_experiment(
                    model_name=model_name,
                    strategy=strategy,
                    X_train=X_train, X_test=X_test,
                    y_train=y_train, y_test=y_test,
                    feature_names=feature_names,
                    task=args.task,
                    label_names=label_names,
                )
                all_results.update(result)
            except Exception as e:
                logger.error(f"Experiment {model_name}_{strategy} failed: {e}")
                continue

    # ── Feature importance (best tree-based model by macro_f1) ────────────
    logger.info("\nComputing feature importance...")
    tree_models = ["random_forest", "hist_gradient_boosting", "xgboost"]
    best_tree_key = None
    best_macro_f1 = -1

    for key, metrics in all_results.items():
        model_name = metrics.get("model_name", "")
        if model_name in tree_models:
            mf1 = metrics.get("macro_f1", 0)
            if mf1 > best_macro_f1:
                best_macro_f1 = mf1
                best_tree_key = key

    if best_tree_key:
        model_path = config.MODELS_DIR / f"{best_tree_key}.pkl"
        if model_path.exists():
            best_model = joblib.load(model_path)
            try:
                imp_df = plot_feature_importance(
                    best_model, feature_names, top_n=20,
                    output_path=config.FIGURES_DIR / f"feature_importance_{best_tree_key}.png"
                )
                logger.info(f"Feature importance computed for: {best_tree_key}")

                # Feature importance for all tree models
                for model_name in tree_models:
                    for strategy in args.strategies:
                        key = f"{model_name}_{strategy}"
                        mp = config.MODELS_DIR / f"{key}.pkl"
                        if mp.exists():
                            m = joblib.load(mp)
                            if hasattr(m, "feature_importances_"):
                                plot_feature_importance(
                                    m, feature_names, top_n=20,
                                    output_path=config.FIGURES_DIR / f"feature_importance_{key}.png"
                                )
                                break
            except Exception as e:
                logger.warning(f"Feature importance failed: {e}")

    # ── Error analysis ────────────────────────────────────────────────────
    logger.info("\nRunning error analysis...")
    for model_name in args.models:
        key = f"{model_name}_none"
        model_path = config.MODELS_DIR / f"{key}.pkl"
        if model_path.exists():
            try:
                m = joblib.load(model_path)
                y_pred = m.predict(X_test)
                error_analysis(
                    y_test, y_pred, label_names,
                    output_dir=config.METRICS_DIR,
                    model_name=key,
                )
            except Exception as e:
                logger.warning(f"Error analysis failed for {key}: {e}")

    # ── Model comparison table ────────────────────────────────────────────
    logger.info("\nGenerating model comparison table...")
    if all_results:
        # Flatten metrics for table
        table_data = {}
        for key, metrics in all_results.items():
            if metrics:
                table_data[key] = metrics
        compare_models(
            table_data,
            config.TABLES_DIR / "model_comparison.csv"
        )

    # ── Print summary ─────────────────────────────────────────────────────
    elapsed = time.time() - t_total
    print(f"\n{'='*70}")
    print(f"EXPERIMENT COMPLETE — {args.task.upper()} — {elapsed:.1f}s total")
    print(f"{'='*70}")
    print(f"{'Model':<45} {'MacroF1':>8} {'ROC-AUC':>8} {'Time(s)':>8}")
    print(f"{'─'*70}")

    sorted_results = sorted(
        [(k, v) for k, v in all_results.items() if v],
        key=lambda x: x[1].get("macro_f1", 0),
        reverse=True
    )
    for key, metrics in sorted_results:
        mf1 = metrics.get("macro_f1", 0)
        auc_ = metrics.get("roc_auc") or "N/A"
        t_ = metrics.get("training_time_s", 0)
        auc_str = f"{auc_:.4f}" if isinstance(auc_, float) else auc_
        print(f"  {key:<43} {mf1:>8.4f} {auc_str:>8} {t_:>8.1f}")

    print(f"{'='*70}\n")
    print(f"Results saved to:")
    print(f"  Metrics : {config.METRICS_DIR}")
    print(f"  Figures : {config.FIGURES_DIR}")
    print(f"  Tables  : {config.TABLES_DIR}")


if __name__ == "__main__":
    main()
