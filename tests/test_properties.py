"""
tests/test_properties.py — Property-based tests using hypothesis.

Tests formal correctness properties from the design document:
- P5: Metric Boundedness
- P6: Determinism
- P7: Feature Dimension Consistency
- P8: Label Encoding Round-Trip
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pytest
import tempfile
import joblib

from hypothesis import given, settings, assume
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
)


# ── Property 5: Metric Boundedness ───────────────────────────────────────────

def compute_metrics_safe(y_true, y_pred, y_prob=None):
    """Compute all metrics, returning dict. Same logic as evaluate.py."""
    results = {}
    results["accuracy"] = accuracy_score(y_true, y_pred)
    results["precision"] = precision_score(y_true, y_pred, zero_division=0)
    results["recall"] = recall_score(y_true, y_pred, zero_division=0)
    results["f1"] = f1_score(y_true, y_pred, zero_division=0)
    results["macro_f1"] = f1_score(y_true, y_pred, average="macro", zero_division=0)
    if y_prob is not None:
        try:
            results["roc_auc"] = roc_auc_score(y_true, y_prob)
        except Exception:
            results["roc_auc"] = None
    return results


@given(
    n=st.integers(10, 100),
    seed=st.integers(0, 2**31),
)
@settings(max_examples=100, deadline=5000)
def test_metric_boundedness(n, seed):
    """
    Property P5: All computed metrics must lie in [0.0, 1.0].
    Applies to any valid binary y_true and y_pred arrays.
    """
    rng = np.random.RandomState(seed)
    # Ensure at least one positive and one negative example
    y_true = np.array([0] + [1] + list(rng.randint(0, 2, n - 2)), dtype=np.int64)
    y_pred = np.array([0] + [1] + list(rng.randint(0, 2, n - 2)), dtype=np.int64)

    metrics = compute_metrics_safe(y_true, y_pred)
    for name, value in metrics.items():
        if value is not None:
            assert 0.0 <= value <= 1.0, (
                f"Metric '{name}' = {value} is outside [0, 1]!"
            )


# ── Property 8: Label Encoding Round-Trip ────────────────────────────────────

@given(
    labels=st.lists(
        st.text(min_size=1, max_size=20,
                alphabet=st.characters(whitelist_categories=('Lu', 'Ll', 'Nd'),
                                       whitelist_characters=' -_')),
        min_size=5, max_size=50, unique=True
    )
)
@settings(max_examples=50, deadline=5000)
def test_label_encoder_round_trip(labels):
    """
    Property P8: Save/load LabelEncoder produces identical transform results.
    """
    assume(len(labels) >= 2)
    le = LabelEncoder()
    le.fit(labels)

    # Encode
    encoded = le.transform(labels)

    # Save and reload
    with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
        path = f.name
    joblib.dump(le, path)
    le_loaded = joblib.load(path)

    # Verify round-trip
    encoded_reloaded = le_loaded.transform(labels)
    np.testing.assert_array_equal(
        encoded, encoded_reloaded,
        err_msg="LabelEncoder round-trip failed: saved/loaded encoder produces different results"
    )

    decoded = le_loaded.inverse_transform(encoded_reloaded)
    assert list(decoded) == list(labels), (
        "LabelEncoder inverse_transform failed after save/load"
    )


# ── Property 7: Feature Dimension Consistency ────────────────────────────────

def test_feature_dimension_consistency():
    """
    Property P7: After fit_transform_pipeline, X_train and X_test
    must have identical feature dimensions.
    """
    import pandas as pd
    from src.preprocessing import (
        clean_data, engineer_targets, split_data,
        build_preprocessing_pipeline, fit_transform_pipeline
    )
    from src import config

    rng = np.random.RandomState(42)
    n = 300
    df = pd.DataFrame({
        "Flow Duration": rng.exponential(1000, n).astype(np.float32),
        "Total Fwd Packets": rng.randint(1, 100, n).astype(np.float32),
        "Flow Bytes/s": rng.uniform(0, 1e6, n).astype(np.float32),
        "Label": rng.choice(["BENIGN", "DoS Hulk", "DDoS"], n, p=[0.7, 0.2, 0.1]),
    })
    df = clean_data(df)
    df, _ = engineer_targets(df)
    X_train, X_test, y_train, y_test = split_data(df, config.BINARY_LABEL_COLUMN)
    pipeline = build_preprocessing_pipeline()
    X_tr, X_te, _ = fit_transform_pipeline(pipeline, X_train, X_test)

    assert X_tr.shape[1] == X_te.shape[1], (
        f"Feature dimension mismatch: train={X_tr.shape[1]}, test={X_te.shape[1]}"
    )
    assert X_tr.shape[1] == X_train.shape[1], (
        f"Feature count changed: before={X_train.shape[1]}, after={X_tr.shape[1]}"
    )


# ── Property 6: Determinism ───────────────────────────────────────────────────

def test_determinism():
    """
    Property P6: Same random_state=42 produces identical splits and scaled arrays.
    """
    import pandas as pd
    from src.preprocessing import (
        clean_data, engineer_targets, split_data,
        build_preprocessing_pipeline, fit_transform_pipeline
    )
    from src import config

    rng = np.random.RandomState(0)
    n = 200
    df = pd.DataFrame({
        "Flow Duration": rng.exponential(1000, n).astype(np.float32),
        "Total Fwd Packets": rng.randint(1, 100, n).astype(np.float32),
        "Flow Bytes/s": rng.uniform(0, 1e6, n).astype(np.float32),
        "Label": rng.choice(["BENIGN", "DoS Hulk"], n, p=[0.7, 0.3]),
    })
    df = clean_data(df)
    df, _ = engineer_targets(df)

    def run_pipeline(df):
        X_train, X_test, y_train, y_test = split_data(
            df, config.BINARY_LABEL_COLUMN, random_state=42
        )
        pipeline = build_preprocessing_pipeline()
        X_tr, X_te, _ = fit_transform_pipeline(pipeline, X_train, X_test)
        return X_tr, X_te, y_train.values, y_test.values

    X_tr1, X_te1, y_tr1, y_te1 = run_pipeline(df)
    X_tr2, X_te2, y_tr2, y_te2 = run_pipeline(df)

    np.testing.assert_array_equal(X_tr1, X_tr2, err_msg="X_train not deterministic")
    np.testing.assert_array_equal(X_te1, X_te2, err_msg="X_test not deterministic")
    np.testing.assert_array_equal(y_tr1, y_tr2, err_msg="y_train not deterministic")
    np.testing.assert_array_equal(y_te1, y_te2, err_msg="y_test not deterministic")
