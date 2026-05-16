"""
PipelineForge Module

Version:
    0.6.0

Updated:
    2026-05-16

Purpose:
    Numeric feature scaling utilities.
"""

from __future__ import annotations

import pandas as pd

from pipelineforge.core.logging import setup_logger
from pipelineforge.exceptions import DataValidationError

logger = setup_logger(__name__)


def scale_numeric_features(
    dataframe: pd.DataFrame,
    columns: list[str],
    strategy: str = "minmax",
) -> pd.DataFrame:
    """
    Scale numeric columns using a specified strategy.

    Parameters
    ----------
    dataframe : pd.DataFrame
    columns : list[str]
        Numeric columns to scale.
    strategy : str
        One of: 'minmax', 'standard'.
        - minmax  : scales to [0, 1]
        - standard: zero mean, unit variance

    Returns
    -------
    pd.DataFrame
    """
    valid_strategies = {"minmax", "standard"}

    if strategy not in valid_strategies:
        raise DataValidationError(
            f"Invalid strategy: '{strategy}'. Must be one of {sorted(valid_strategies)}"
        )

    missing = [col for col in columns if col not in dataframe.columns]
    if missing:
        raise DataValidationError(f"Columns not found for scaling: {missing}")

    non_numeric = [col for col in columns if not pd.api.types.is_numeric_dtype(dataframe[col])]
    if non_numeric:
        raise DataValidationError(f"Non-numeric columns cannot be scaled: {non_numeric}")

    result = dataframe.copy()

    for col in columns:
        if strategy == "minmax":
            col_min = result[col].min()
            col_max = result[col].max()

            if col_min == col_max:
                raise DataValidationError(
                    f"Cannot apply minmax scaling to '{col}': min == max ({col_min})"
                )

            result[col] = (result[col] - col_min) / (col_max - col_min)

        elif strategy == "standard":
            col_mean = result[col].mean()
            col_std = result[col].std()

            if col_std == 0:
                raise DataValidationError(f"Cannot apply standard scaling to '{col}': std == 0")

            result[col] = (result[col] - col_mean) / col_std

        logger.info("scale_numeric_features: scaled '%s' using strategy='%s'", col, strategy)

    return result
