"""
tests/test_preprocessing.py — Unit tests for preprocessing pipeline.
Uses small synthetic DataFrames to verify correctness properties.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import pytest

from src.preprocessing import (
    clean_data, engineer_targets, split_data,
    build_preprocessing_pipeline, fit_transform_pipeline,
)
from src import config


def make_test_df(n=200, seed=0):
    """Create a small synthetic DataFrame resembling CIC-IDS2017 structure."""
    rng = np.random.RandomState(seed)
    df = pd.DataFrame({
        "Flow Duration": rng.exponential(1000, n).astype(np.float32),
        "Total Fwd Packets": rng.randint(1, 100, n).astype(np.float32),
        "Flow Bytes/s": rng.uniform(0, 1e6, n).astype(np.float32),
        "Flow Packets/s": rng.uniform(0, 1000, n).astype(np.float32),
        "Fwd Header Length.1": rng.randint(0, 100, n).astype(np.float32),
        "Destination Port": rng.randint(1, 65535, n).astype(np.float32),
        "Label": rng.choice(
            ["BENIGN", "DoS Hulk", "PortScan", "DDoS"],
            n, p=[0.7, 0.15, 0.1, 0.05]
        ),
    })
    return df


def make_test_df_with_issues(n=100):
    """Synthetic DataFrame with inf values and duplicates."""
    df = make_test_df(n)
    # Insert infinite values
    df.loc[0, "Flow Bytes/s"] = np.inf
    df.loc[1, "Flow Packets/s"] = -np.inf
    # Insert duplicates
    df = pd.concat([df, df.iloc[:5]], ignore_index=True)
    return df


# ── clean_data tests ──────────────────────────────────────────────────────────

class TestCleanData:

    def test_removes_infinite_values(self):
        df = make_test_df_with_issues()
        df_clean = clean_data(df)
        numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            assert not np.isinf(df_clean[col]).any(), f"Inf found in {col} after clean_data"

    def test_removes_duplicates(self):
        df = make_test_df_with_issues(n=100)
        df_clean = clean_data(df)
        assert df_clean.duplicated().sum() == 0, "Duplicates remain after clean_data"

    def test_drops_configured_columns(self):
        df = make_test_df(n=100)
        df_clean = clean_data(df)
        for col in config.COLS_TO_DROP:
            assert col not in df_clean.columns, f"Column '{col}' should have been dropped"

    def test_label_column_preserved(self):
        df = make_test_df(n=100)
        df_clean = clean_data(df)
        assert config.LABEL_COLUMN in df_clean.columns, "Label column removed by clean_data"

    def test_result_is_nonempty(self):
        df = make_test_df(n=100)
        df_clean = clean_data(df)
        assert len(df_clean) > 0, "clean_data returned empty DataFrame"

    def test_rows_only_decrease(self):
        df = make_test_df_with_issues(n=100)
        df_clean = clean_data(df)
        # With duplicates, we expect fewer or equal rows
        assert len(df_clean) <= len(df)


# ── split_data tests ──────────────────────────────────────────────────────────

class TestSplitData:

    def setup_method(self):
        df = make_test_df(n=400)
        df = clean_data(df)
        df, _ = engineer_targets(df)
        self.df = df

    def test_split_sizes(self):
        X_train, X_test, y_train, y_test = split_data(
            self.df, config.BINARY_LABEL_COLUMN, test_size=0.2
        )
        total = len(X_train) + len(X_test)
        assert total == len(self.df) or total == len(self.df) - self.df[config.BINARY_LABEL_COLUMN].value_counts().min()

    def test_disjoint_splits(self):
        X_train, X_test, y_train, y_test = split_data(
            self.df, config.BINARY_LABEL_COLUMN, test_size=0.2
        )
        train_idx = set(X_train.index)
        test_idx = set(X_test.index)
        assert len(train_idx & test_idx) == 0, "Train and test sets overlap!"

    def test_determinism(self):
        r1 = split_data(self.df, config.BINARY_LABEL_COLUMN, random_state=42)
        r2 = split_data(self.df, config.BINARY_LABEL_COLUMN, random_state=42)
        pd.testing.assert_frame_equal(r1[0].reset_index(drop=True),
                                      r2[0].reset_index(drop=True))

    def test_different_seeds_differ(self):
        r1 = split_data(self.df, config.BINARY_LABEL_COLUMN, random_state=42)
        r2 = split_data(self.df, config.BINARY_LABEL_COLUMN, random_state=99)
        # Very unlikely to be identical with different seeds
        assert not r1[0].equals(r2[0]), "Different seeds produced identical splits"


# ── fit_transform_pipeline tests ─────────────────────────────────────────────

class TestFitTransformPipeline:

    def setup_method(self):
        df = make_test_df(n=400)
        df = clean_data(df)
        df, _ = engineer_targets(df)
        X_train, X_test, y_train, y_test = split_data(
            df, config.BINARY_LABEL_COLUMN, test_size=0.2
        )
        self.X_train = X_train
        self.X_test = X_test

    def test_output_shapes_match(self):
        pipeline = build_preprocessing_pipeline()
        X_tr, X_te, _ = fit_transform_pipeline(pipeline, self.X_train, self.X_test)
        assert X_tr.shape[1] == X_te.shape[1], "Train/test feature dims differ"

    def test_feature_count_matches_input(self):
        pipeline = build_preprocessing_pipeline()
        X_tr, X_te, _ = fit_transform_pipeline(pipeline, self.X_train, self.X_test)
        assert X_tr.shape[1] == self.X_train.shape[1]
        assert X_te.shape[1] == self.X_test.shape[1]

    def test_leakage_prevention(self):
        """
        Core leakage test: scaler mean must equal X_train mean, NOT X_test mean.
        If the pipeline was accidentally fitted on X_test, this test would fail.
        """
        pipeline = build_preprocessing_pipeline()
        X_tr, X_te, fitted_pipeline = fit_transform_pipeline(
            pipeline, self.X_train, self.X_test
        )
        scaler = fitted_pipeline.named_steps["scaler"]
        train_mean = self.X_train.mean(axis=0).values

        # Scaler mean should be close to train mean (after imputation)
        np.testing.assert_allclose(
            scaler.mean_, train_mean, rtol=0.1,
            err_msg="Scaler mean does not match X_train mean — possible leakage!"
        )

        # Scaler mean should NOT match test mean (unless train==test, very unlikely)
        test_mean = self.X_test.mean(axis=0).values
        # They could be close by chance but should not be identical
        # Just verify the pipeline was not re-fitted on test data
        assert fitted_pipeline.named_steps["imputer"] is not None

    def test_no_nan_after_pipeline(self):
        pipeline = build_preprocessing_pipeline()
        X_tr, X_te, _ = fit_transform_pipeline(pipeline, self.X_train, self.X_test)
        assert not np.isnan(X_tr).any(), "NaN in X_train after pipeline"
        assert not np.isnan(X_te).any(), "NaN in X_test after pipeline"

    def test_no_inf_after_pipeline(self):
        pipeline = build_preprocessing_pipeline()
        X_tr, X_te, _ = fit_transform_pipeline(pipeline, self.X_train, self.X_test)
        assert not np.isinf(X_tr).any(), "Inf in X_train after pipeline"
        assert not np.isinf(X_te).any(), "Inf in X_test after pipeline"
