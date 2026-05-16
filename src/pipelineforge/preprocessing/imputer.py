"""
PipelineForge Module

Version:
    0.6.0

Updated:
    2026-05-16

Purpose:
    Missing value imputation utilities.
"""

from __future__ import annotations

import pandas as pd

from pipelineforge.core.logging import setup_logger
from pipelineforge.exceptions import DataValidationError

logger = setup_logger(__name__)


def fill_missing_values(
    dataframe: pd.DataFrame,
    strategy: str,
    columns: list[str] | None = None,
    fill_value: object = None,
) -> pd.DataFrame:
    """
    Fill missing values using a specified strategy.

    Parameters
    ----------
    dataframe : pd.DataFrame
    strategy : str
        One of: 'mean', 'median', 'mode', 'constant'.
    columns : list[str] | None
        Columns to impute. All columns if None.
    fill_value : object
        Value to use when strategy is 'constant'.

    Returns
    -------
    pd.DataFrame
    """
    valid_strategies = {"mean", "median", "mode", "constant"}

    if strategy not in valid_strategies:
        raise DataValidationError(
            f"Invalid strategy: '{strategy}'. Must be one of {sorted(valid_strategies)}"
        )

    if strategy == "constant" and fill_value is None:
        raise DataValidationError("fill_value must be provided when strategy is 'constant'")

    result = dataframe.copy()

    target_cols = columns if columns is not None else result.columns.tolist()

    for col in target_cols:
        if col not in result.columns:
            raise DataValidationError(f"Column not found for imputation: '{col}'")

        null_count = result[col].isnull().sum()

        if null_count == 0:
            continue

        if strategy == "mean":
            if not pd.api.types.is_numeric_dtype(result[col]):
                raise DataValidationError(
                    f"Strategy 'mean' requires numeric column, "
                    f"got '{result[col].dtype}' for '{col}'"
                )
            result[col] = result[col].fillna(result[col].mean())

        elif strategy == "median":
            if not pd.api.types.is_numeric_dtype(result[col]):
                raise DataValidationError(
                    f"Strategy 'median' requires numeric column, "
                    f"got '{result[col].dtype}' for '{col}'"
                )
            result[col] = result[col].fillna(result[col].median())

        elif strategy == "mode":
            mode_val = result[col].mode()
            if mode_val.empty:
                raise DataValidationError(f"Cannot compute mode for column '{col}'")
            result[col] = result[col].fillna(mode_val.iloc[0])

        elif strategy == "constant":
            result[col] = result[col].fillna(fill_value)

        logger.info(
            "fill_missing_values: filled %d null(s) in '%s' using strategy='%s'",
            null_count,
            col,
            strategy,
        )

    return result
