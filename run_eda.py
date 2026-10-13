"""
run_eda.py — Phase 6: Exploratory Data Analysis
Generates all required EDA figures from the processed data.
Saves figures to results/figures/.
"""

import sys
import gc
from pathlib import Path
from collections import Counter

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from src import config

# Style settings
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({"figure.dpi": config.FIGURE_DPI, "font.size": 11})

config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def save_fig(name):
    path = config.FIGURES_DIR / f"{name}.{config.FIGURE_FORMAT}"
    plt.savefig(path, dpi=config.FIGURE_DPI, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path.name}")


def load_processed():
    """Load pre-processed arrays from data/processed/."""
    X_train = np.load(config.PROCESSED_DATA_DIR / "X_train.npy")
    X_test  = np.load(config.PROCESSED_DATA_DIR / "X_test.npy")
    y_train = np.load(config.PROCESSED_DATA_DIR / "y_train_binary.npy")
    y_test  = np.load(config.PROCESSED_DATA_DIR / "y_test_binary.npy")
    with open(config.PROCESSED_DATA_DIR / "feature_names.txt") as f:
        feature_names = [l.strip() for l in f.readlines()]
    return X_train, X_test, y_train, y_test, feature_names


def collect_label_stats():
    """Collect label statistics by scanning CSV files (batch mode)."""
    from src.preprocessing import clean_data
    label_counter = Counter()
    csv_files = sorted(config.RAW_DATA_DIR.glob("*.csv"))
    for f in csv_files:
        df = pd.read_csv(f, low_memory=False, usecols=lambda c: c.strip() in ["Label", " Label"])
        df.columns = df.columns.str.strip()
        if "Label" in df.columns:
            label_counter.update(df["Label"].value_counts().to_dict())
        del df
        gc.collect()
    return label_counter


def main():
    print("\n" + "="*60)
    print("PHASE 6: EXPLORATORY DATA ANALYSIS")
    print("="*60)

    # Load processed arrays
    print("\nLoading processed data...")
    X_train, X_test, y_train, y_test, feature_names = load_processed()
    X_all = np.vstack([X_train, X_test])
    y_all = np.concatenate([y_train, y_test])
    print(f"  X: {X_all.shape}, y binary: {y_all.shape}")

    # ── Figure 1: Binary class distribution ───────────────────────────────
    print("\n[1] Binary class distribution...")
    fig, ax = plt.subplots(figsize=(7, 5))
    unique, counts = np.unique(y_all, return_counts=True)
    labels = ["BENIGN", "ATTACK"]
    colors = ["#2196F3", "#F44336"]
    bars = ax.bar(labels, counts, color=colors, edgecolor="white", linewidth=1.5)
    for bar, cnt in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 500,
                f"{cnt:,}\n({100*cnt/len(y_all):.1f}%)",
                ha="center", va="bottom", fontsize=12, fontweight="bold")
    ax.set_title("Binary Class Distribution (200K-row stratified subset)",
                 fontsize=14, fontweight="bold")
    ax.set_ylabel("Number of Samples")
    ax.set_ylim(0, counts.max() * 1.15)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{int(x):,}"))
    plt.tight_layout()
    save_fig("class_distribution_binary")

    # ── Figure 2: Multiclass distribution ─────────────────────────────────
    print("\n[2] Multiclass attack distribution (scanning CSVs)...")
    label_counter = collect_label_stats()
    total = sum(label_counter.values())
    sorted_labels = sorted(label_counter.items(), key=lambda x: -x[1])

    fig, ax = plt.subplots(figsize=(14, 6))
    names = [l[:30] for l, _ in sorted_labels]
    vals = [c for _, c in sorted_labels]
    colors_mc = ["#2196F3" if n == "BENIGN" else "#F44336" for n in names]
    bars = ax.bar(range(len(names)), vals, color=colors_mc, edgecolor="white")
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=9)
    ax.set_yscale("log")
    ax.set_title("Attack Category Distribution (CIC-IDS2017, log scale)",
                 fontsize=14, fontweight="bold")
    ax.set_ylabel("Number of Samples (log scale)")
    for bar, cnt in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.1,
                f"{cnt:,}", ha="center", va="bottom", fontsize=7, rotation=45)
    plt.tight_layout()
    save_fig("class_distribution_multiclass")

    # ── Figure 3: Feature distributions (BENIGN vs ATTACK) ────────────────
    print("\n[3] Feature distributions BENIGN vs ATTACK...")
    target_features = [
        "Flow Duration", "Flow Bytes/s", "Packet Length Mean",
        "Flow IAT Mean", "SYN Flag Count", "FIN Flag Count",
    ]
    present = [f for f in target_features if f in feature_names]
    df_plot = pd.DataFrame(X_all, columns=feature_names)
    df_plot["Label"] = ["BENIGN" if y == 0 else "ATTACK" for y in y_all]

    fig, axes = plt.subplots(2, 3, figsize=(16, 8))
    axes = axes.flatten()
    for i, feat in enumerate(present[:6]):
        ax = axes[i]
        for label, color in [("BENIGN", "#2196F3"), ("ATTACK", "#F44336")]:
            vals = df_plot.loc[df_plot["Label"] == label, feat].dropna()
            # Clip to 1st–99th percentile for readability
            lo, hi = np.percentile(vals, [1, 99])
            vals_clipped = vals.clip(lo, hi)
            ax.hist(vals_clipped, bins=50, alpha=0.6, color=color,
                    label=label, density=True)
        ax.set_title(feat, fontsize=10, fontweight="bold")
        ax.set_xlabel("Value (1st–99th percentile)")
        ax.set_ylabel("Density")
        ax.legend(fontsize=8)
    plt.suptitle("Feature Distributions: BENIGN vs ATTACK",
                 fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    save_fig("feature_distributions")

    # ── Figure 4: Correlation heatmap ─────────────────────────────────────
    print("\n[4] Correlation heatmap (top 20 variance features)...")
    df_feat = pd.DataFrame(X_all, columns=feature_names)
    # Top 20 by variance
    variances = df_feat.var()
    top20 = variances.nlargest(20).index.tolist()
    corr = df_feat[top20].corr()

    fig, ax = plt.subplots(figsize=(14, 11))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=False, cmap="coolwarm",
                center=0, vmin=-1, vmax=1, ax=ax,
                linewidths=0.5, cbar_kws={"shrink": 0.8})
    ax.set_title("Pearson Correlation — Top 20 High-Variance Features",
                 fontsize=13, fontweight="bold")
    plt.xticks(rotation=45, ha="right", fontsize=8)
    plt.yticks(fontsize=8)
    plt.tight_layout()
    save_fig("correlation_heatmap")

    # ── Figure 5: Class imbalance visualization ────────────────────────────
    print("\n[5] Class imbalance visualization...")
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Left: pie chart
    ax = axes[0]
    n_benign = (y_all == 0).sum()
    n_attack = (y_all == 1).sum()
    ax.pie([n_benign, n_attack], labels=["BENIGN", "ATTACK"],
           colors=["#2196F3", "#F44336"],
           autopct="%1.1f%%", startangle=90,
           wedgeprops={"edgecolor": "white", "linewidth": 2})
    ax.set_title("Binary Class Proportions\n(200K subset)", fontweight="bold")

    # Right: log-scale bar of all classes
    ax2 = axes[1]
    names2 = [l[:25] for l, _ in sorted_labels]
    vals2 = [c for _, c in sorted_labels]
    colors2 = ["#2196F3" if n == "BENIGN" else "#F44336" for n in names2]
    ax2.barh(range(len(names2)), vals2, color=colors2)
    ax2.set_yticks(range(len(names2)))
    ax2.set_yticklabels(names2, fontsize=8)
    ax2.set_xscale("log")
    ax2.set_xlabel("Number of Samples (log scale)")
    ax2.set_title("All 15 Classes — Imbalance Ratio", fontweight="bold")
    ax2.axvline(x=np.median(vals2), color="gray", linestyle="--",
                alpha=0.5, label="median")
    ax2.legend(fontsize=9)
    plt.suptitle("Class Imbalance in CIC-IDS2017",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    save_fig("class_imbalance")

    print(f"\n{'='*60}")
    print(f"EDA COMPLETE — all figures saved to results/figures/")
    print(f"{'='*60}\n")

    # Print EDA observations summary
    imbalance_ratio = n_benign / max(n_attack, 1)
    dominant = sorted_labels[1][0] if len(sorted_labels) > 1 else "N/A"
    rarest  = sorted_labels[-1][0] if sorted_labels else "N/A"

    print("KEY OBSERVATIONS:")
    print(f"  Imbalance ratio (BENIGN:ATTACK) : {imbalance_ratio:.1f}:1")
    print(f"  Dominant attack type             : {dominant} ({sorted_labels[1][1]:,} samples)")
    print(f"  Rarest attack type               : {rarest} ({sorted_labels[-1][1]:,} samples)")
    print(f"  Total classes                    : {len(label_counter)}")
    print(f"  Features analyzed                : {len(feature_names)}")


if __name__ == "__main__":
    main()
