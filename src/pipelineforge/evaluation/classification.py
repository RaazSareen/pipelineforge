"""
PipelineForge Module

Version:
    0.9.0

Updated:
    2026-05-16

Purpose:
    Classification evaluation metrics.
    Accepts y_true and y_pred directly — decoupled from model objects.
    Use model.predict() before passing results here.
"""

from __future__ import annotations

import math
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from pipelineforge.core.logging import setup_logger
from pipelineforge.exceptions import DataValidationError

logger = setup_logger(__name__)

_VALID_AVERAGE = {"weighted", "macro", "micro"}


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


def _validate_average(average: str) -> None:
    if average not in _VALID_AVERAGE:
        raise DataValidationError(
            f"Invalid average: '{average}'. Must be one of {sorted(_VALID_AVERAGE)}"
        )


def compute_accuracy(
    y_true: pd.Series | np.ndarray,
    y_pred: pd.Series | np.ndarray,
) -> float:
    """
    Compute accuracy score.

    Parameters
    ----------
    y_true : pd.Series | np.ndarray
    y_pred : pd.Series | np.ndarray

    Returns
    -------
    float
    """
    _validate_inputs(y_true, y_pred)
    score = float(accuracy_score(y_true, y_pred))
    logger.info("compute_accuracy: %.4f", score)
    return score


def compute_precision(
    y_true: pd.Series | np.ndarray,
    y_pred: pd.Series | np.ndarray,
    average: str = "weighted",
) -> float:
    """
    Compute precision score.

    Parameters
    ----------
    y_true : pd.Series | np.ndarray
    y_pred : pd.Series | np.ndarray
    average : str
        One of: 'weighted', 'macro', 'micro'.

    Returns
    -------
    float
    """
    _validate_inputs(y_true, y_pred)
    _validate_average(average)
    score = float(precision_score(y_true, y_pred, average=average, zero_division=0))
    logger.info("compute_precision: %.4f (average=%s)", score, average)
    return score


def compute_recall(
    y_true: pd.Series | np.ndarray,
    y_pred: pd.Series | np.ndarray,
    average: str = "weighted",
) -> float:
    """
    Compute recall score.

    Parameters
    ----------
    y_true : pd.Series | np.ndarray
    y_pred : pd.Series | np.ndarray
    average : str
        One of: 'weighted', 'macro', 'micro'.

    Returns
    -------
    float
    """
    _validate_inputs(y_true, y_pred)
    _validate_average(average)
    score = float(recall_score(y_true, y_pred, average=average, zero_division=0))
    logger.info("compute_recall: %.4f (average=%s)", score, average)
    return score


def compute_f1(
    y_true: pd.Series | np.ndarray,
    y_pred: pd.Series | np.ndarray,
    average: str = "weighted",
) -> float:
    """
    Compute F1 score.

    Parameters
    ----------
    y_true : pd.Series | np.ndarray
    y_pred : pd.Series | np.ndarray
    average : str
        One of: 'weighted', 'macro', 'micro'.

    Returns
    -------
    float
    """
    _validate_inputs(y_true, y_pred)
    _validate_average(average)
    score = float(f1_score(y_true, y_pred, average=average, zero_division=0))
    logger.info("compute_f1: %.4f (average=%s)", score, average)
    return score


def compute_roc_auc(
    y_true: pd.Series | np.ndarray,
    y_score: pd.Series | np.ndarray,
    average: str = "weighted",
    multi_class: str = "ovr",
) -> float:
    """
    Compute ROC AUC score from predicted probabilities.

    Parameters
    ----------
    y_true : pd.Series | np.ndarray
        True class labels.
    y_score : pd.Series | np.ndarray
        Predicted probabilities from model.predict_proba().
        For binary: 1D array of positive class probabilities.
        For multiclass: 2D array of shape (n_samples, n_classes).
    average : str
        One of: 'weighted', 'macro', 'micro'.
    multi_class : str
        One of: 'ovr', 'ovo'. Used for multiclass problems.

    Returns
    -------
    float
    """
    _validate_average(average)

    if len(y_true) == 0:
        raise DataValidationError("y_true is empty")
    if len(y_score) == 0:
        raise DataValidationError("y_score is empty")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            score = float(
                roc_auc_score(
                    y_true,
                    y_score,
                    average=average,
                    multi_class=multi_class,
                )
            )
        except ValueError as exc:
            raise DataValidationError(f"ROC AUC computation failed: {exc}") from exc

    if math.isnan(score):
        raise DataValidationError("ROC AUC is undefined: y_true contains only one class")

    logger.info("compute_roc_auc: %.4f", score)
    return score


def compute_confusion_matrix(
    y_true: pd.Series | np.ndarray,
    y_pred: pd.Series | np.ndarray,
) -> pd.DataFrame:
    """
    Compute confusion matrix as a labeled DataFrame.

    Parameters
    ----------
    y_true : pd.Series | np.ndarray
    y_pred : pd.Series | np.ndarray

    Returns
    -------
    pd.DataFrame
        Rows = actual classes, columns = predicted classes.
    """
    _validate_inputs(y_true, y_pred)

    labels = sorted(set(y_true))
    matrix = confusion_matrix(y_true, y_pred, labels=labels)

    result = pd.DataFrame(
        matrix,
        index=[f"actual_{label}" for label in labels],
        columns=[f"predicted_{label}" for label in labels],
    )

    logger.info("compute_confusion_matrix: shape=%s", result.shape)
    return result


def compute_classification_report(
    y_true: pd.Series | np.ndarray,
    y_pred: pd.Series | np.ndarray,
) -> str:
    """
    Compute sklearn classification report as a string.

    Parameters
    ----------
    y_true : pd.Series | np.ndarray
    y_pred : pd.Series | np.ndarray

    Returns
    -------
    str
    """
    _validate_inputs(y_true, y_pred)
    report = classification_report(y_true, y_pred, zero_division=0)
    logger.info("compute_classification_report: generated")
    return report


def compute_all_classification_metrics(
    y_true: pd.Series | np.ndarray,
    y_pred: pd.Series | np.ndarray,
    average: str = "weighted",
) -> dict[str, float]:
    """
    Compute all scalar classification metrics in one call.

    Does not include confusion matrix or classification report.

    Parameters
    ----------
    y_true : pd.Series | np.ndarray
    y_pred : pd.Series | np.ndarray
    average : str
        One of: 'weighted', 'macro', 'micro'.

    Returns
    -------
    dict[str, float]
        Keys: accuracy, precision, recall, f1.
    """
    _validate_inputs(y_true, y_pred)
    _validate_average(average)

    metrics = {
        "accuracy": compute_accuracy(y_true, y_pred),
        "precision": compute_precision(y_true, y_pred, average=average),
        "recall": compute_recall(y_true, y_pred, average=average),
        "f1": compute_f1(y_true, y_pred, average=average),
    }

    logger.info("compute_all_classification_metrics: %s", metrics)
    return metrics
