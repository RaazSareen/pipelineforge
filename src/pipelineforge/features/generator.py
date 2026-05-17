"""
PipelineForge Module

Version:
    0.7.0

Updated:
    2026-05-16

Purpose:
    Derived feature generation utilities.
    Creates new columns from existing ones: ratios, differences, aggregates.
    Does not apply statistical tests or use model outputs.
"""

from __future__ import annotations

import pandas as pd

from pipelineforge.core.logging import setup_logger
from pipelineforge.exceptions import DataValidationError

logger = setup_logger(__name__)


def add_ratio_feature(
    dataframe: pd.DataFrame,
    numerator: str,
    denominator: str,
    output_column: str,
    fill_value: float = 0.0,
) -> pd.DataFrame:
    """
    Add a ratio feature: numerator / denominator.

    Division by zero is replaced with fill_value.

    Parameters
    ----------
    dataframe : pd.DataFrame
    numerator : str
        Column to use as numerator.
    denominator : str
        Column to use as denominator.
    output_column : str
        Name of the new column.
    fill_value : float
        Value to use where denominator is zero. Default 0.0.

    Returns
    -------
    pd.DataFrame
    """
    for col in (numerator, denominator):
        if col not in dataframe.columns:
            raise DataValidationError(f"Column not found: '{col}'")
        if not pd.api.types.is_numeric_dtype(dataframe[col]):
            raise DataValidationError(f"Column '{col}' must be numeric for ratio feature")

    result = dataframe.copy()

    result[output_column] = result[numerator] / result[denominator].replace(0, float("nan"))
    result[output_column] = result[output_column].fillna(fill_value)

    logger.info(
        "add_ratio_feature: created '%s' = %s / %s",
        output_column,
        numerator,
        denominator,
    )

    return result


def add_difference_feature(
    dataframe: pd.DataFrame,
    col_a: str,
    col_b: str,
    output_column: str,
) -> pd.DataFrame:
    """
    Add a difference feature: col_a - col_b.

    Parameters
    ----------
    dataframe : pd.DataFrame
    col_a : str
    col_b : str
    output_column : str
        Name of the new column.

    Returns
    -------
    pd.DataFrame
    """
    for col in (col_a, col_b):
        if col not in dataframe.columns:
            raise DataValidationError(f"Column not found: '{col}'")
        if not pd.api.types.is_numeric_dtype(dataframe[col]):
            raise DataValidationError(f"Column '{col}' must be numeric for difference feature")

    result = dataframe.copy()
    result[output_column] = result[col_a] - result[col_b]

    logger.info(
        "add_difference_feature: created '%s' = %s - %s",
        output_column,
        col_a,
        col_b,
    )

    return result


def add_aggregate_feature(
    dataframe: pd.DataFrame,
    columns: list[str],
    operation: str,
    output_column: str,
) -> pd.DataFrame:
    """
    Add a row-wise aggregate feature across specified columns.

    Parameters
    ----------
    dataframe : pd.DataFrame
    columns : list[str]
        Numeric columns to aggregate.
    operation : str
        One of: 'sum', 'mean', 'min', 'max'.
    output_column : str
        Name of the new column.

    Returns
    -------
    pd.DataFrame
    """
    valid_operations = {"sum", "mean", "min", "max"}

    if operation not in valid_operations:
        raise DataValidationError(
            f"Invalid operation: '{operation}'. Must be one of {sorted(valid_operations)}"
        )

    missing = [col for col in columns if col not in dataframe.columns]
    if missing:
        raise DataValidationError(f"Columns not found: {missing}")

    non_numeric = [col for col in columns if not pd.api.types.is_numeric_dtype(dataframe[col])]
    if non_numeric:
        raise DataValidationError(f"Non-numeric columns cannot be aggregated: {non_numeric}")

    result = dataframe.copy()

    agg_func = getattr(result[columns], operation)
    result[output_column] = agg_func(axis=1)

    logger.info(
        "add_aggregate_feature: created '%s' = %s(%s)",
        output_column,
        operation,
        columns,
    )

    return result
