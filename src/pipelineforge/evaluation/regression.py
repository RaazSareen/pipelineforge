"""
PipelineForge Module

Version:
    0.9.0

Updated:
    2026-05-16

Purpose:
    Regression evaluation metrics.
    Accepts y_true and y_pred directly — decoupled from model objects.
    Use model.predict() before passing results here.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score,
)

from pipelineforge.core.logging import setup_logger
from pipelineforge.exceptions import DataValidationError

logger = setup_logger(__name__)


def _validate_inputs(
    y_true: pd.Series | np.ndarray,
    y_pred: pd.Series | np.ndarray,
) -> None:
    if len(y_true) == 0:
        raise DataValidationError("y_true is empty")
    if len(y_pred) == 0:
        raise DataValidationError("y_pred is empty")
    if len(y_true) != len(y_pred):
        raise DataValidationError(
            f"y_true and y_pred must have the same length. "
            f"Got y_true={len(y_true)}, y_pred={len(y_pred)}"
        )


def compute_mae(
    y_true: pd.Series | np.ndarray,
    y_pred: pd.Series | np.ndarray,
) -> float:
    """
    Compute Mean Absolute Error.

    Parameters
    ----------
    y_true : pd.Series | np.ndarray
    y_pred : pd.Series | np.ndarray

    Returns
    -------
    float
    """
    _validate_inputs(y_true, y_pred)
    score = float(mean_absolute_error(y_true, y_pred))
    logger.info("compute_mae: %.4f", score)
    return score


def compute_mse(
    y_true: pd.Series | np.ndarray,
    y_pred: pd.Series | np.ndarray,
) -> float:
    """
    Compute Mean Squared Error.

    Parameters
    ----------
    y_true : pd.Series | np.ndarray
    y_pred : pd.Series | np.ndarray

    Returns
    -------
    float
    """
    _validate_inputs(y_true, y_pred)
    score = float(mean_squared_error(y_true, y_pred))
    logger.info("compute_mse: %.4f", score)
    return score


def compute_rmse(
    y_true: pd.Series | np.ndarray,
    y_pred: pd.Series | np.ndarray,
) -> float:
    """
    Compute Root Mean Squared Error.

    Parameters
    ----------
    y_true : pd.Series | np.ndarray
    y_pred : pd.Series | np.ndarray

    Returns
    -------
    float
    """
    _validate_inputs(y_true, y_pred)
    score = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    logger.info("compute_rmse: %.4f", score)
    return score


def compute_r2(
    y_true: pd.Series | np.ndarray,
    y_pred: pd.Series | np.ndarray,
) -> float:
    """
    Compute R-squared score.

    Parameters
    ----------
    y_true : pd.Series | np.ndarray
    y_pred : pd.Series | np.ndarray

    Returns
    -------
    float
    """
    _validate_inputs(y_true, y_pred)
    score = float(r2_score(y_true, y_pred))
    logger.info("compute_r2: %.4f", score)
    return score


def compute_mape(
    y_true: pd.Series | np.ndarray,
    y_pred: pd.Series | np.ndarray,
) -> float:
    """
    Compute Mean Absolute Percentage Error.

    Parameters
    ----------
    y_true : pd.Series | np.ndarray
    y_pred : pd.Series | np.ndarray

    Returns
    -------
    float
    """
    _validate_inputs(y_true, y_pred)

    y_true_arr = np.asarray(y_true)

    if (y_true_arr == 0).any():
        raise DataValidationError("MAPE is undefined when y_true contains zero values")

    score = float(mean_absolute_percentage_error(y_true, y_pred))
    logger.info("compute_mape: %.4f", score)
    return score


def compute_all_regression_metrics(
    y_true: pd.Series | np.ndarray,
    y_pred: pd.Series | np.ndarray,
) -> dict[str, float]:
    """
    Compute all regression metrics in one call.

    Parameters
    ----------
    y_true : pd.Series | np.ndarray
    y_pred : pd.Series | np.ndarray

    Returns
    -------
    dict[str, float]
        Keys: mae, mse, rmse, r2, mape.
        mape is omitted if y_true contains zeros.
    """
    _validate_inputs(y_true, y_pred)

    metrics: dict[str, float] = {
        "mae": compute_mae(y_true, y_pred),
        "mse": compute_mse(y_true, y_pred),
        "rmse": compute_rmse(y_true, y_pred),
        "r2": compute_r2(y_true, y_pred),
    }

    try:
        metrics["mape"] = compute_mape(y_true, y_pred)
    except DataValidationError:
        logger.warning("compute_all_regression_metrics: mape skipped (y_true contains zeros)")

    logger.info("compute_all_regression_metrics: %s", metrics)
    return metrics
