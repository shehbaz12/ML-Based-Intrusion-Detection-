"""
feature_analysis.py — Feature importance, correlation analysis, and SHAP.

Supports RF, HGB, XGBoost via model.feature_importances_.
SHAP is optional — gracefully skipped if not installed or blocked.
"""

import logging
from pathlib import Path
from typing import List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src import config

logger = logging.getLogger(__name__)


def plot_feature_importance(
    model,
    feature_names: List[str],
    top_n: int = 20,
    output_path: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Extract feature_importances_ and plot top_n features as horizontal bar chart.

    Returns DataFrame with columns ['feature', 'importance'] sorted descending.
    Also saves CSV alongside the PNG if output_path is provided.
    """
    if not hasattr(model, "feature_importances_"):
        raise AttributeError(
            f"{type(model).__name__} does not have feature_importances_. "
            "Use a tree-based model (RF, HGB, XGBoost)."
        )

    importances = model.feature_importances_
    df = pd.DataFrame({
        "feature": feature_names,
        "importance": importances,
    }).sort_values("importance", ascending=False).reset_index(drop=True)

    top_df = df.head(top_n)

    fig, ax = plt.subplots(figsize=(10, max(5, top_n * 0.35)))
    colors = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(top_df)))
    ax.barh(top_df["feature"][::-1], top_df["importance"][::-1],
            color=colors[::-1])
    ax.set_xlabel("Feature Importance (Gini/Gain)", fontsize=11)
    ax.set_title(f"Top {top_n} Feature Importances — {type(model).__name__}",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=config.FIGURE_DPI, bbox_inches="tight")
        # Save CSV alongside
        csv_path = output_path.with_suffix(".csv")
        df.to_csv(csv_path, index=False)
        logger.info(f"Feature importance saved: {output_path.name}, {csv_path.name}")
    plt.close()

    return df


def plot_correlation_matrix(
    df: pd.DataFrame,
    output_path: Path,
    top_n: int = 25,
) -> None:
    """
    Compute Pearson correlation on top_n highest-variance features.
    Save heatmap as PNG.
    """
    numeric_df = df.select_dtypes(include=[np.number])
    variances = numeric_df.var()
    top_cols = variances.nlargest(top_n).index.tolist()
    corr = numeric_df[top_cols].corr()

    fig, ax = plt.subplots(figsize=(13, 11))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=False, cmap="coolwarm",
                center=0, vmin=-1, vmax=1, ax=ax,
                linewidths=0.3, cbar_kws={"shrink": 0.8})
    ax.set_title(f"Pearson Correlation — Top {top_n} High-Variance Features",
                 fontsize=13, fontweight="bold")
    plt.xticks(rotation=45, ha="right", fontsize=8)
    plt.yticks(fontsize=8)
    plt.tight_layout()
    plt.savefig(output_path, dpi=config.FIGURE_DPI, bbox_inches="tight")
    plt.close()
    logger.info(f"Correlation matrix saved: {Path(output_path).name}")


def compute_shap_values(
    model,
    X_sample: np.ndarray,
    feature_names: List[str],
    output_dir: Path,
    max_samples: int = 1000,
) -> Optional[np.ndarray]:
    """
    Compute SHAP values using TreeExplainer (tree-based models only).
    Returns SHAP values array or None if SHAP is unavailable.

    Saves summary plot and beeswarm plot to output_dir.
    Computation limited to max_samples rows for CPU performance.
    """
    try:
        import shap
    except ImportError:
        logger.warning(
            "SHAP not installed or blocked. Skipping SHAP analysis. "
            "Feature importance from feature_importances_ is still available."
        )
        return None

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Subsample for performance
    n = min(len(X_sample), max_samples)
    idx = np.random.RandomState(config.RANDOM_STATE).choice(
        len(X_sample), n, replace=False
    )
    X_sub = X_sample[idx]
    logger.info(f"Computing SHAP values on {n} samples...")

    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_sub)

        # Summary plot
        fig, ax = plt.subplots(figsize=(10, 8))
        shap.summary_plot(shap_values, X_sub,
                          feature_names=feature_names,
                          show=False, max_display=20)
        plt.title(f"SHAP Summary — {type(model).__name__}", fontweight="bold")
        plt.tight_layout()
        plt.savefig(output_dir / "shap_summary.png",
                    dpi=config.FIGURE_DPI, bbox_inches="tight")
        plt.close()
        logger.info("SHAP summary plot saved.")

        return shap_values

    except Exception as e:
        logger.warning(f"SHAP computation failed: {e}. Skipping.")
        return None


def select_features_by_importance(
    model,
    feature_names: List[str],
    threshold: float = 0.01,
) -> List[str]:
    """
    Return feature names with importance >= threshold.
    Used for reduced-feature experiment variants.

    Postcondition: all returned features have importance >= threshold,
    no feature with importance < threshold is included.
    """
    if not hasattr(model, "feature_importances_"):
        raise AttributeError(
            f"{type(model).__name__} does not have feature_importances_."
        )

    importances = model.feature_importances_
    selected = [
        name for name, imp in zip(feature_names, importances)
        if imp >= threshold
    ]

    logger.info(
        f"Feature selection (threshold={threshold}): "
        f"{len(selected)}/{len(feature_names)} features selected"
    )
    return selected
