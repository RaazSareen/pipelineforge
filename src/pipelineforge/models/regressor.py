"""
PipelineForge Module

Version:
    0.8.0

Updated:
    2026-05-16

Purpose:
    Regression model registry, instantiation, and fitting.
    Accepts pre-split data — use model_selection.splitter for splitting.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge

from pipelineforge.core.logging import setup_logger
from pipelineforge.exceptions import DataValidationError, PipelineExecutionError

logger = setup_logger(__name__)

_REGRESSOR_REGISTRY: dict[str, type[BaseEstimator]] = {
    "random_forest": RandomForestRegressor,
    "linear_regression": LinearRegression,
    "gradient_boosting": GradientBoostingRegressor,
    "ridge": Ridge,
}


def get_available_regressors() -> list[str]:
    """Return sorted list of supported regressor keys."""
    return sorted(_REGRESSOR_REGISTRY.keys())


def build_regressor(model_type: str, **kwargs: Any) -> BaseEstimator:
    """
    Instantiate a regressor by registry key.

    Parameters
    ----------
    model_type : str
        One of the supported regressor keys.
    **kwargs
        Passed directly to the regressor constructor.

    Returns
    -------
    BaseEstimator
    """
    if model_type not in _REGRESSOR_REGISTRY:
        raise DataValidationError(
            f"Unknown regressor: '{model_type}'. Available: {get_available_regressors()}"
        )

    model = _REGRESSOR_REGISTRY[model_type](**kwargs)

    logger.info("build_regressor: instantiated '%s'", model_type)

    return model


def train_regressor(
    model: BaseEstimator,
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> BaseEstimator:
    """
    Fit a regressor on pre-split training data.

    Parameters
    ----------
    model : BaseEstimator
        Unfitted sklearn regressor.
    X_train : pd.DataFrame
        Training features.
    y_train : pd.Series
        Training target values.

    Returns
    -------
    BaseEstimator
        Fitted regressor.
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

    logger.info("train_regressor: fitting on %d samples", len(X_train))

    try:
        model.fit(X_train, y_train)
    except Exception as exc:
        raise PipelineExecutionError(f"Regressor training failed: {exc}") from exc

    logger.info("train_regressor: training complete")

    return model
