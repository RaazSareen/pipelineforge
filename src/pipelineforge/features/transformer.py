"""
PipelineForge Module

Version:
    0.7.0

Updated:
    2026-05-16

Purpose:
    Mathematical and structural feature transformations.
    Includes log, polynomial, binning, and datetime decomposition.
    Does not generate interactions or apply statistical selection.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from pipelineforge.core.logging import setup_logger
from pipelineforge.exceptions import DataValidationError

logger = setup_logger(__name__)


def apply_log_transform(
    dataframe: pd.DataFrame,
    columns: list[str],
    output_suffix: str = "_log",
) -> pd.DataFrame:
    """
    Apply log1p transformation to numeric columns.

    Uses log1p (log(1 + x)) to handle zero values safely.

    Parameters
    ----------
    dataframe : pd.DataFrame
    columns : list[str]
        Numeric columns to transform.
    output_suffix : str
        Suffix appended to original column name for new column.

    Returns
    -------
    pd.DataFrame
    """
    missing = [col for col in columns if col not in dataframe.columns]
    if missing:
        raise DataValidationError(f"Columns not found: {missing}")

    non_numeric = [col for col in columns if not pd.api.types.is_numeric_dtype(dataframe[col])]
    if non_numeric:
        raise DataValidationError(f"Non-numeric columns cannot be log-transformed: {non_numeric}")

    negative = [col for col in columns if (dataframe[col] < 0).any()]
    if negative:
        raise DataValidationError(
            f"Columns with negative values cannot be log-transformed: {negative}"
        )

    result = dataframe.copy()

    for col in columns:
        result[f"{col}{output_suffix}"] = np.log1p(result[col])
        logger.info("apply_log_transform: created '%s%s'", col, output_suffix)

    return result


def apply_polynomial_features(
    dataframe: pd.DataFrame,
    columns: list[str],
    degree: int = 2,
) -> pd.DataFrame:
    """
    Add polynomial features (powers) for specified columns.

    For each column and each power from 2 to degree,
    adds a new column named '{col}_pow{power}'.

    Parameters
    ----------
    dataframe : pd.DataFrame
    columns : list[str]
        Numeric columns to expand.
    degree : int
        Maximum polynomial degree. Must be >= 2.

    Returns
    -------
    pd.DataFrame
    """
    if degree < 2:
        raise DataValidationError(f"degree must be >= 2, got {degree}")

    missing = [col for col in columns if col not in dataframe.columns]
    if missing:
        raise DataValidationError(f"Columns not found: {missing}")

    non_numeric = [col for col in columns if not pd.api.types.is_numeric_dtype(dataframe[col])]
    if non_numeric:
        raise DataValidationError(f"Non-numeric columns cannot be expanded: {non_numeric}")

    result = dataframe.copy()

    for col in columns:
        for power in range(2, degree + 1):
            result[f"{col}_pow{power}"] = result[col] ** power
            logger.info("apply_polynomial_features: created '%s_pow%d'", col, power)

    return result


def apply_binning(
    dataframe: pd.DataFrame,
    column: str,
    bins: int | list[float],
    output_column: str,
    labels: list[str] | None = None,
) -> pd.DataFrame:
    """
    Bin a numeric column into discrete intervals.

    Parameters
    ----------
    dataframe : pd.DataFrame
    column : str
        Numeric column to bin.
    bins : int | list[float]
        Number of equal-width bins or explicit bin edges.
    output_column : str
        Name of the new binned column.
    labels : list[str] | None
        Labels for the bins. If None, uses integer codes.

    Returns
    -------
    pd.DataFrame
    """
    if column not in dataframe.columns:
        raise DataValidationError(f"Column not found: '{column}'")

    if not pd.api.types.is_numeric_dtype(dataframe[column]):
        raise DataValidationError(f"Column '{column}' must be numeric for binning")

    result = dataframe.copy()

    try:
        result[output_column] = pd.cut(
            result[column],
            bins=bins,
            labels=labels,
            include_lowest=True,
        )
    except ValueError as exc:
        raise DataValidationError(f"Binning failed: {exc}") from exc

    logger.info("apply_binning: created '%s' from '%s'", output_column, column)

    return result


def extract_datetime_features(
    dataframe: pd.DataFrame,
    column: str,
    features: list[str] | None = None,
) -> pd.DataFrame:
    """
    Extract datetime components from a datetime column.

    Parameters
    ----------
    dataframe : pd.DataFrame
    column : str
        Datetime column to decompose.
    features : list[str] | None
        Components to extract. Supported:
        'year', 'month', 'day', 'hour', 'minute',
        'dayofweek', 'quarter', 'weekofyear'.
        Extracts all if None.

    Returns
    -------
    pd.DataFrame
    """
    if column not in dataframe.columns:
        raise DataValidationError(f"Column not found: '{column}'")

    valid_features = {
        "year",
        "month",
        "day",
        "hour",
        "minute",
        "dayofweek",
        "quarter",
        "weekofyear",
    }

    target_features = features if features is not None else sorted(valid_features)

    invalid = [f for f in target_features if f not in valid_features]
    if invalid:
        raise DataValidationError(
            f"Invalid datetime features: {invalid}. Supported: {sorted(valid_features)}"
        )

    result = dataframe.copy()

    try:
        dt = pd.to_datetime(result[column])
    except Exception as exc:
        raise DataValidationError(f"Cannot parse column '{column}' as datetime: {exc}") from exc

    feature_map = {
        "year": dt.dt.year,
        "month": dt.dt.month,
        "day": dt.dt.day,
        "hour": dt.dt.hour,
        "minute": dt.dt.minute,
        "dayofweek": dt.dt.dayofweek,
        "quarter": dt.dt.quarter,
        "weekofyear": dt.dt.isocalendar().week.astype(int),
    }

    for feat in target_features:
        result[f"{column}_{feat}"] = feature_map[feat]
        logger.info("extract_datetime_features: created '%s_%s'", column, feat)

    return result
