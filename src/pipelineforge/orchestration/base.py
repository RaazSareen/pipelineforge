"""
PipelineForge Module

Version:
    0.11.0

Updated:
    2026-05-19

Purpose:
    Shared result dataclass for orchestration pipelines.
    PipelineResult is returned by all run_*_pipeline functions.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.base import BaseEstimator


@dataclass(slots=True)
class PipelineResult:
    """
    Standardized result container for all pipeline executions.

    Attributes
    ----------
    model : BaseEstimator
        Fitted sklearn estimator.
    metrics : dict[str, float]
        Evaluation metrics from the pipeline run.
    X_train : pd.DataFrame
        Training features.
    X_test : pd.DataFrame
        Test features.
    y_train : pd.Series
        Training target.
    y_test : pd.Series
        Test target.
    predictions : pd.Series
        Model predictions on X_test, index-aligned with y_test.
    model_type : str
        Registry key used to build the model.
    target_column : str
        Name of the target column used in this run.
    """

    model: BaseEstimator
    metrics: dict[str, float]
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    predictions: pd.Series
    model_type: str
    target_column: str
