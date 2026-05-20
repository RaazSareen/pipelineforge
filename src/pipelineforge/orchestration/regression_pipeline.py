"""
PipelineForge Module

Version:
    0.11.0

Updated:
    2026-05-19

Purpose:
    End-to-end regression pipeline.
    Assumes data is already cleaned, encoded, and scaled.
    Handles: split → build → train → evaluate → return PipelineResult.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from pipelineforge.core.logging import setup_logger
from pipelineforge.evaluation.regression import compute_all_regression_metrics
from pipelineforge.exceptions import DataValidationError
from pipelineforge.model_selection.splitter import train_test_split_data
from pipelineforge.models.regressor import build_regressor, train_regressor
from pipelineforge.orchestration.base import PipelineResult

logger = setup_logger(__name__)


def run_regression_pipeline(
    dataframe: pd.DataFrame,
    target_column: str,
    model_type: str = "linear_regression",
    test_size: float = 0.2,
    random_state: int = 42,
    model_kwargs: dict[str, Any] | None = None,
) -> PipelineResult:
    """
    Run an end-to-end regression pipeline.

    Assumes input data is already preprocessed.
    Steps: validate → split → build → train → evaluate → return result.

    Parameters
    ----------
    dataframe : pd.DataFrame
        Clean, encoded, scaled DataFrame ready for modeling.
    target_column : str
        Name of the target column.
    model_type : str
        Regressor registry key. Default: 'linear_regression'.
    test_size : float
        Fraction of data for test set.
    random_state : int
        Random seed for reproducibility.
    model_kwargs : dict | None
        Additional kwargs passed to the regressor constructor.

    Returns
    -------
    PipelineResult
    """
    if dataframe.empty:
        raise DataValidationError("dataframe is empty")

    if target_column not in dataframe.columns:
        raise DataValidationError(f"Target column not found: '{target_column}'")

    kwargs = model_kwargs or {}

    logger.info(
        "run_regression_pipeline: model=%s | target=%s | test_size=%.2f",
        model_type,
        target_column,
        test_size,
    )

    X_train, X_test, y_train, y_test = train_test_split_data(
        dataframe,
        target_column=target_column,
        test_size=test_size,
        random_state=random_state,
    )

    model = build_regressor(model_type, **kwargs)
    model = train_regressor(model, X_train, y_train)

    y_pred = model.predict(X_test)
    predictions = pd.Series(y_pred, index=y_test.index, name="predictions")

    metrics = compute_all_regression_metrics(y_test, predictions)

    logger.info("run_regression_pipeline: metrics=%s", metrics)

    return PipelineResult(
        model=model,
        metrics=metrics,
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        predictions=predictions,
        model_type=model_type,
        target_column=target_column,
    )
