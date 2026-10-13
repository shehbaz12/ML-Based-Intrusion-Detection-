"""
evaluate.py — Metrics computation, figure generation, and results persistence.

All figures saved at 300 DPI. No interactive display (Agg backend).
All metrics saved as JSON for reproducible audit.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score,
    confusion_matrix, classification_report,
    roc_curve, precision_recall_curve, auc,
)

from src import config

logger = logging.getLogger(__name__)

sns.set_theme(style="whitegrid")


# ── Metrics ───────────────────────────────────────────────────────────────────

def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None,
    average: str = "binary",
) -> Dict[str, Any]:
    """
    Compute evaluation metrics.

    Returns dict with:
      accuracy, precision, recall, f1, macro_f1,
      roc_auc (if y_prob provided), pr_auc (if y_prob provided)

    All numeric values guaranteed in [0.0, 1.0].
    """
    results: Dict[str, Any] = {}

    results["accuracy"]  = float(accuracy_score(y_true, y_pred))
    results["precision"] = float(precision_score(y_true, y_pred, average=average,
                                                  zero_division=0))
    results["recall"]    = float(recall_score(y_true, y_pred, average=average,
                                              zero_division=0))
    results["f1"]        = float(f1_score(y_true, y_pred, average=average,
                                          zero_division=0))
    results["macro_f1"]  = float(f1_score(y_true, y_pred, average="macro",
                                          zero_division=0))

    if y_prob is not None:
        try:
            if average == "binary":
                probs = y_prob[:, 1] if y_prob.ndim == 2 else y_prob
                results["roc_auc"] = float(roc_auc_score(y_true, probs))
                results["pr_auc"]  = float(average_precision_score(y_true, probs))
            else:
                results["roc_auc"] = float(
                    roc_auc_score(y_true, y_prob, multi_class="ovr",
                                  average="macro")
                )
                results["pr_auc"] = None
        except Exception as e:
            logger.warning(f"Could not compute ROC/PR AUC: {e}")
            results["roc_auc"] = None
            results["pr_auc"]  = None
    else:
        results["roc_auc"] = None
        results["pr_auc"]  = None

    # Postcondition: all numeric values in [0, 1]
    for k, v in results.items():
        if isinstance(v, float):
            assert 0.0 <= v <= 1.0, f"Metric '{k}' = {v} outside [0,1]"

    return results


# ── Figures ───────────────────────────────────────────────────────────────────

def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str],
    output_path: Path,
    title: str = "Confusion Matrix",
) -> None:
    """Save normalized confusion matrix heatmap at 300 DPI."""
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Raw counts
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names,
                ax=axes[0])
    axes[0].set_title(f"{title}\n(Raw Counts)", fontweight="bold")
    axes[0].set_xlabel("Predicted")
    axes[0].set_ylabel("True")

    # Normalized
    sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names,
                ax=axes[1], vmin=0, vmax=1)
    axes[1].set_title(f"{title}\n(Normalized)", fontweight="bold")
    axes[1].set_xlabel("Predicted")
    axes[1].set_ylabel("True")

    plt.tight_layout()
    plt.savefig(output_path, dpi=config.FIGURE_DPI, bbox_inches="tight")
    plt.close()
    logger.info(f"Confusion matrix saved: {Path(output_path).name}")


def plot_roc_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    model_name: str,
    output_path: Path,
) -> None:
    """Save ROC curve with AUC annotation at 300 DPI."""
    probs = y_prob[:, 1] if y_prob.ndim == 2 else y_prob
    fpr, tpr, _ = roc_curve(y_true, probs)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(fpr, tpr, color="#2196F3", lw=2,
            label=f"ROC curve (AUC = {roc_auc:.4f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5, label="Random classifier")
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title(f"ROC Curve — {model_name}", fontsize=13, fontweight="bold")
    ax.legend(fontsize=11)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])
    plt.tight_layout()
    plt.savefig(output_path, dpi=config.FIGURE_DPI, bbox_inches="tight")
    plt.close()
    logger.info(f"ROC curve saved: {Path(output_path).name}")


def plot_pr_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    model_name: str,
    output_path: Path,
) -> None:
    """Save Precision-Recall curve with PR-AUC annotation at 300 DPI."""
    probs = y_prob[:, 1] if y_prob.ndim == 2 else y_prob
    precision, recall, _ = precision_recall_curve(y_true, probs)
    pr_auc = auc(recall, precision)
    baseline = y_true.mean()

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(recall, precision, color="#F44336", lw=2,
            label=f"PR curve (AUC = {pr_auc:.4f})")
    ax.axhline(y=baseline, color="gray", linestyle="--", lw=1,
               label=f"Baseline (prevalence = {baseline:.3f})")
    ax.set_xlabel("Recall", fontsize=12)
    ax.set_ylabel("Precision", fontsize=12)
    ax.set_title(f"Precision-Recall Curve — {model_name}",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=11)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])
    plt.tight_layout()
    plt.savefig(output_path, dpi=config.FIGURE_DPI, bbox_inches="tight")
    plt.close()
    logger.info(f"PR curve saved: {Path(output_path).name}")


# ── Persistence ───────────────────────────────────────────────────────────────

def save_metrics(
    metrics: Dict[str, Any],
    model_name: str,
    output_dir: Path,
) -> Path:
    """Save metrics dict to JSON at output_dir/{model_name}_metrics.json."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{model_name}_metrics.json"

    # Convert numpy types to native Python for JSON serialization
    clean = {}
    for k, v in metrics.items():
        if isinstance(v, (np.integer, np.int64)):
            clean[k] = int(v)
        elif isinstance(v, (np.floating, np.float64)):
            clean[k] = float(v)
        elif isinstance(v, np.ndarray):
            clean[k] = v.tolist()
        else:
            clean[k] = v

    with open(path, "w") as f:
        json.dump(clean, f, indent=2)
    logger.info(f"Metrics saved: {path.name}")
    return path


def compare_models(
    results: Dict[str, Dict[str, Any]],
    output_path: Path,
) -> pd.DataFrame:
    """
    Build comparison DataFrame from results dict, save as CSV and PNG table.

    Args:
        results: {experiment_key: metrics_dict}
        output_path: path for CSV output (PNG saved alongside)

    Returns:
        Sorted comparison DataFrame
    """
    rows = []
    for key, metrics in results.items():
        row = {"experiment": key}
        row.update(metrics)
        rows.append(row)

    df = pd.DataFrame(rows)

    # Sort by macro_f1 descending
    if "macro_f1" in df.columns:
        df = df.sort_values("macro_f1", ascending=False).reset_index(drop=True)

    # Save CSV
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info(f"Comparison CSV saved: {output_path.name}")

    # Save formatted PNG table
    png_path = output_path.with_suffix(".png")
    _save_table_figure(df, png_path)

    return df


def _save_table_figure(df: pd.DataFrame, path: Path) -> None:
    """Save DataFrame as a formatted table figure."""
    # Select display columns
    display_cols = ["experiment", "accuracy", "precision", "recall",
                    "f1", "macro_f1", "roc_auc", "training_time_s"]
    cols = [c for c in display_cols if c in df.columns]
    df_disp = df[cols].copy()

    # Format floats
    for col in cols[1:]:
        if df_disp[col].dtype == float:
            df_disp[col] = df_disp[col].apply(
                lambda x: f"{x:.4f}" if x is not None and not pd.isna(x) else "N/A"
            )

    fig_h = max(3, len(df_disp) * 0.5 + 1)
    fig, ax = plt.subplots(figsize=(16, fig_h))
    ax.axis("off")
    table = ax.table(
        cellText=df_disp.values,
        colLabels=df_disp.columns,
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.auto_set_column_width(col=list(range(len(df_disp.columns))))

    # Header styling
    for j in range(len(df_disp.columns)):
        table[0, j].set_facecolor("#1565C0")
        table[0, j].set_text_props(color="white", fontweight="bold")

    # Alternating row colors
    for i in range(1, len(df_disp) + 1):
        color = "#E3F2FD" if i % 2 == 0 else "white"
        for j in range(len(df_disp.columns)):
            table[i, j].set_facecolor(color)

    plt.title("Model Comparison — Sorted by Macro F1",
              fontsize=12, fontweight="bold", pad=10)
    plt.tight_layout()
    plt.savefig(path, dpi=config.FIGURE_DPI, bbox_inches="tight")
    plt.close()
    logger.info(f"Comparison table figure saved: {path.name}")


# ── Error Analysis ────────────────────────────────────────────────────────────

def error_analysis(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    label_names: List[str],
    output_dir: Path,
    model_name: str = "model",
) -> Dict[str, Any]:
    """
    Systematic error analysis:
    - Count FP and FN (binary task)
    - Per-class recall (multiclass task)
    - Identify hard-to-detect classes (recall < median)
    - Save figure and JSON summary

    Returns summary dict.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    n_classes = len(np.unique(y_true))
    is_binary = n_classes == 2

    summary: Dict[str, Any] = {"model_name": model_name}

    if is_binary:
        # Binary FP/FN
        tn = int(((y_pred == 0) & (y_true == 0)).sum())
        fp = int(((y_pred == 1) & (y_true == 0)).sum())
        fn = int(((y_pred == 0) & (y_true == 1)).sum())
        tp = int(((y_pred == 1) & (y_true == 1)).sum())

        summary["true_positives"]  = tp
        summary["true_negatives"]  = tn
        summary["false_positives"] = fp
        summary["false_negatives"] = fn
        summary["fp_rate"] = round(fp / max(fp + tn, 1), 4)
        summary["fn_rate"] = round(fn / max(fn + tp, 1), 4)

        print(f"\n  Error Analysis ({model_name}, binary):")
        print(f"    TP: {tp:,}  TN: {tn:,}  FP: {fp:,}  FN: {fn:,}")
        print(f"    FP rate: {summary['fp_rate']:.4f}  FN rate: {summary['fn_rate']:.4f}")

    else:
        # Multiclass per-class recall
        cr = classification_report(y_true, y_pred, target_names=label_names,
                                   output_dict=True, zero_division=0)
        per_class_recall = {}
        for cls in label_names:
            if cls in cr:
                per_class_recall[cls] = round(cr[cls]["recall"], 4)

        recalls = list(per_class_recall.values())
        median_recall = float(np.median(recalls))
        hard_classes = [cls for cls, r in per_class_recall.items()
                        if r < median_recall]

        summary["per_class_recall"] = per_class_recall
        summary["median_recall"]    = round(median_recall, 4)
        summary["hard_to_detect"]   = hard_classes

        print(f"\n  Error Analysis ({model_name}, multiclass):")
        print(f"    Median per-class recall: {median_recall:.4f}")
        print(f"    Hard-to-detect classes ({len(hard_classes)}): {hard_classes}")

        # Plot per-class recall bar chart
        fig, ax = plt.subplots(figsize=(12, 5))
        cls_list = list(per_class_recall.keys())
        rec_list = list(per_class_recall.values())
        colors = ["#F44336" if r < median_recall else "#4CAF50" for r in rec_list]
        ax.barh(cls_list, rec_list, color=colors)
        ax.axvline(x=median_recall, color="navy", linestyle="--",
                   label=f"Median recall: {median_recall:.3f}")
        ax.set_xlabel("Recall")
        ax.set_title(f"Per-Class Recall — {model_name}\n"
                     f"(Red = below median, hard-to-detect)",
                     fontweight="bold")
        ax.legend()
        plt.tight_layout()
        fig_path = output_dir / f"per_class_recall_{model_name}.png"
        plt.savefig(fig_path, dpi=config.FIGURE_DPI, bbox_inches="tight")
        plt.close()
        logger.info(f"Per-class recall figure saved: {fig_path.name}")

    # Save summary JSON
    summary_path = output_dir / f"error_analysis_{model_name}.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Error analysis saved: {summary_path.name}")

    return summary
