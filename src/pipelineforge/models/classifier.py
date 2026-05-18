"""
PipelineForge Module

Version:
    0.8.0

Updated:
    2026-05-16

Purpose:
    Classification model registry, instantiation, and fitting.
    Accepts pre-split data — use model_selection.splitter for splitting.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from pipelineforge.core.logging import setup_logger
from pipelineforge.exceptions import DataValidationError, PipelineExecutionError

logger = setup_logger(__name__)

_CLASSIFIER_REGISTRY: dict[str, type[BaseEstimator]] = {
    "random_forest": RandomForestClassifier,
    "logistic_regression": LogisticRegression,
    "gradient_boosting": GradientBoostingClassifier,
}


def get_available_classifiers() -> list[str]:
    """Return sorted list of supported classifier keys."""
    return sorted(_CLASSIFIER_REGISTRY.keys())


def build_classifier(model_type: str, **kwargs: Any) -> BaseEstimator:
    """
    Instantiate a classifier by registry key.

    Parameters
    ----------
    model_type : str
        One of the supported classifier keys.
    **kwargs
        Passed directly to the classifier constructor.

    Returns
    -------
    BaseEstimator
    """
    if model_type not in _CLASSIFIER_REGISTRY:
        raise DataValidationError(
            f"Unknown classifier: '{model_type}'. Available: {get_available_classifiers()}"
        )

    model = _CLASSIFIER_REGISTRY[model_type](**kwargs)

    logger.info("build_classifier: instantiated '%s'", model_type)

    return model


def train_classifier(
    model: BaseEstimator,
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> BaseEstimator:
    """
    Fit a classifier on pre-split training data.

    Parameters
    ----------
    model : BaseEstimator
        Unfitted sklearn classifier.
    X_train : pd.DataFrame
        Training features.
    y_train : pd.Series
        Training labels.

    Returns
    -------
    BaseEstimator
        Fitted classifier.
    """
    if X_train.empty:
        raise DataValidationError("X_train is empty")

    if y_train.empty:
        raise DataValidationError("y_train is empty")

    if len(X_train) != len(y_train):
        raise DataValidationError(
            f"X_train and y_train must have the same length. "
            f"Got X_train={len(X_train)}, y_train={len(y_train)}"
        )

    logger.info(
        "train_classifier: fitting on %d samples | classes=%s",
        len(X_train),
        sorted(y_train.unique().tolist()),
    )

    try:
        model.fit(X_train, y_train)
    except Exception as exc:
        raise PipelineExecutionError(f"Classifier training failed: {exc}") from exc

    logger.info("train_classifier: training complete")

    return model
