"""
train.py — Model training, persistence, and cross-validation.

Supports four model families:
  - logistic_regression
  - random_forest
  - hist_gradient_boosting
  - xgboost  (CPU-only: tree_method="hist")

All hyperparameters are sourced from config.MODEL_CONFIGS.
"""

import logging
import time
from pathlib import Path
from typing import Optional, Tuple

import joblib
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score

from src import config

logger = logging.getLogger(__name__)


def get_model(
    model_name: str,
    class_weight: Optional[str] = None,
    random_state: int = config.RANDOM_STATE,
) -> BaseEstimator:
    """
    Return a configured but unfitted estimator.

    Args:
        model_name: one of 'logistic_regression', 'random_forest',
                    'hist_gradient_boosting', 'xgboost'
        class_weight: 'balanced' or None (ignored for hist_gradient_boosting)
        random_state: integer seed

    Returns:
        Unfitted sklearn-compatible estimator
    """
    params = config.MODEL_CONFIGS.get(model_name, {}).copy()
    params["random_state"] = random_state

    if model_name == "logistic_regression":
        if class_weight:
            params["class_weight"] = class_weight
        return LogisticRegression(**params)

    elif model_name == "random_forest":
        if class_weight:
            params["class_weight"] = class_weight
        return RandomForestClassifier(**params)

    elif model_name == "hist_gradient_boosting":
        # HGB does not support class_weight parameter directly
        # class imbalance handled via sample_weight or scoring
        return HistGradientBoostingClassifier(**params)

    elif model_name == "xgboost":
        try:
            from xgboost import XGBClassifier
        except ImportError:
            raise ImportError("xgboost is not installed. Run: pip install xgboost")
        if class_weight == "balanced":
            # XGBoost uses scale_pos_weight for binary, not class_weight
            logger.warning(
                "XGBoost does not support class_weight='balanced'. "
                "Use scale_pos_weight for binary tasks. Ignoring class_weight."
            )
        return XGBClassifier(**params)

    else:
        raise ValueError(
            f"Unknown model: '{model_name}'. "
            f"Choose from: {list(config.MODEL_CONFIGS.keys())}"
        )


def train_model(
    model: BaseEstimator,
    X_train: np.ndarray,
    y_train: np.ndarray,
) -> Tuple[BaseEstimator, float]:
    """
    Fit model on training data and return (fitted_model, training_time_seconds).
    """
    model_name = type(model).__name__
    logger.info(f"Training {model_name} on {X_train.shape[0]:,} samples...")

    start = time.time()
    model.fit(X_train, y_train)
    elapsed = time.time() - start

    logger.info(f"  {model_name} trained in {elapsed:.1f}s")
    return model, elapsed


def save_model(model: BaseEstimator, path: Path) -> None:
    """Serialize fitted model to disk using joblib."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    logger.info(f"Model saved: {path.name}")


def load_model(path: Path) -> BaseEstimator:
    """Load a previously saved model from disk."""
    model = joblib.load(path)
    logger.info(f"Model loaded: {Path(path).name}")
    return model


def run_cross_validation(
    model: BaseEstimator,
    X_train: np.ndarray,
    y_train: np.ndarray,
    cv: int = config.CV_FOLDS,
    scoring: str = "f1_macro",
) -> dict:
    """
    Stratified k-fold cross-validation on training data ONLY.
    Never touches the test set.

    Returns:
        dict with 'mean', 'std', 'scores'
    """
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=config.RANDOM_STATE)
    scores = cross_val_score(
        model, X_train, y_train, cv=skf, scoring=scoring, n_jobs=-1
    )
    result = {
        "mean": float(scores.mean()),
        "std": float(scores.std()),
        "scores": scores.tolist(),
        "scoring": scoring,
        "cv_folds": cv,
    }
    logger.info(
        f"CV ({cv}-fold, {scoring}): {result['mean']:.4f} ± {result['std']:.4f}"
    )
    return result
