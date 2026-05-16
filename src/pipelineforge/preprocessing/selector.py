"""
PipelineForge Module

Version:
    0.6.0

Updated:
    2026-05-16

Purpose:
    Feature selection utilities.
"""

from __future__ import annotations

import pandas as pd

from pipelineforge.core.logging import setup_logger
from pipelineforge.exceptions import DataValidationError

logger = setup_logger(__name__)


def drop_columns(
    dataframe: pd.DataFrame,
    columns: list[str],
) -> pd.DataFrame:
    """
    Drop specified columns from the DataFrame.

    Parameters
    ----------
    dataframe : pd.DataFrame
    columns : list[str]
        Column names to drop.

    Returns
    -------
    pd.DataFrame
    """
    missing = [col for col in columns if col not in dataframe.columns]

    if missing:
        raise DataValidationError(f"Columns not found for drop: {missing}")

    result = dataframe.drop(columns=columns)

    logger.info("drop_columns: dropped %d column(s): %s", len(columns), columns)

    return result


def select_columns(
    dataframe: pd.DataFrame,
    columns: list[str],
) -> pd.DataFrame:
    """
    Select and return only specified columns.

    Parameters
    ----------
    dataframe : pd.DataFrame
    columns : list[str]
        Column names to keep.

    Returns
    -------
    pd.DataFrame
    """
    missing = [col for col in columns if col not in dataframe.columns]

    if missing:
        raise DataValidationError(f"Columns not found for selection: {missing}")

    result = dataframe[columns].copy()

    logger.info("select_columns: selected %d column(s): %s", len(columns), columns)

    return result


def drop_low_variance(
    dataframe: pd.DataFrame,
    threshold: float = 0.0,
    columns: list[str] | None = None,
) -> pd.DataFrame:
    """
    Drop numeric columns whose variance is at or below the threshold.

    Parameters
    ----------
    dataframe : pd.DataFrame
    threshold : float
        Variance at or below this value triggers drop. Default is 0.0.
    columns : list[str] | None
        Columns to evaluate. Auto-selects numeric columns if None.

    Returns
    -------
    pd.DataFrame
    """
    if threshold < 0:
        raise DataValidationError(f"threshold must be >= 0, got {threshold}")

    target_cols = (
        columns
        if columns is not None
        else dataframe.select_dtypes(include="number").columns.tolist()
    )

    missing = [col for col in target_cols if col not in dataframe.columns]
    if missing:
        raise DataValidationError(f"Columns not found: {missing}")

    to_drop = [col for col in target_cols if dataframe[col].var() <= threshold]

    if to_drop:
        logger.info(
            "drop_low_variance: dropping %d low-variance column(s): %s",
            len(to_drop),
            to_drop,
        )

    return dataframe.drop(columns=to_drop)
