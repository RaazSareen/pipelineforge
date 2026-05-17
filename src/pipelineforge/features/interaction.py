"""
PipelineForge Module

Version:
    0.7.0

Updated:
    2026-05-16

Purpose:
    Pairwise and multiplicative feature interaction utilities.
    Creates new columns from combinations of existing numeric features.
    Does not apply statistical tests or use model outputs.
"""

from __future__ import annotations

from itertools import combinations

import pandas as pd

from pipelineforge.core.logging import setup_logger
from pipelineforge.exceptions import DataValidationError

logger = setup_logger(__name__)


def add_interaction_features(
    dataframe: pd.DataFrame,
    columns: list[str],
    separator: str = "_x_",
) -> pd.DataFrame:
    """
    Add pairwise multiplicative interaction features.

    For each pair (col_a, col_b) in columns, creates:
        col_a_x_col_b = col_a * col_b

    Parameters
    ----------
    dataframe : pd.DataFrame
    columns : list[str]
        Numeric columns to interact. Minimum 2 required.
    separator : str
        String placed between column names in output column name.

    Returns
    -------
    pd.DataFrame
    """
    if len(columns) < 2:
        raise DataValidationError("At least 2 columns are required for interaction features")

    missing = [col for col in columns if col not in dataframe.columns]
    if missing:
        raise DataValidationError(f"Columns not found: {missing}")

    non_numeric = [col for col in columns if not pd.api.types.is_numeric_dtype(dataframe[col])]
    if non_numeric:
        raise DataValidationError(
            f"Non-numeric columns cannot be used for interactions: {non_numeric}"
        )

    result = dataframe.copy()

    for col_a, col_b in combinations(columns, 2):
        output_col = f"{col_a}{separator}{col_b}"
        result[output_col] = result[col_a] * result[col_b]
        logger.info("add_interaction_features: created '%s'", output_col)

    return result


def add_division_interaction(
    dataframe: pd.DataFrame,
    numerator: str,
    denominator: str,
    output_column: str,
    fill_value: float = 0.0,
) -> pd.DataFrame:
    """
    Add a single division-based interaction feature.

    Distinct from generator.add_ratio_feature in intent:
    this captures a directional relationship between two
    features as part of feature engineering, not data preparation.

    Division by zero is replaced with fill_value.

    Parameters
    ----------
    dataframe : pd.DataFrame
    numerator : str
    denominator : str
    output_column : str
    fill_value : float

    Returns
    -------
    pd.DataFrame
    """
    for col in (numerator, denominator):
        if col not in dataframe.columns:
            raise DataValidationError(f"Column not found: '{col}'")
        if not pd.api.types.is_numeric_dtype(dataframe[col]):
            raise DataValidationError(f"Column '{col}' must be numeric for division interaction")

    result = dataframe.copy()

    result[output_column] = result[numerator] / result[denominator].replace(0, float("nan"))
    result[output_column] = result[output_column].fillna(fill_value)

    logger.info(
        "add_division_interaction: created '%s' = %s / %s",
        output_column,
        numerator,
        denominator,
    )

    return result


def add_polynomial_interaction(
    dataframe: pd.DataFrame,
    col_a: str,
    col_b: str,
    output_column: str,
    power_a: int = 1,
    power_b: int = 1,
) -> pd.DataFrame:
    """
    Add a polynomial interaction: col_a^power_a * col_b^power_b.

    Parameters
    ----------
    dataframe : pd.DataFrame
    col_a : str
    col_b : str
    output_column : str
    power_a : int
        Power applied to col_a. Default 1.
    power_b : int
        Power applied to col_b. Default 1.

    Returns
    -------
    pd.DataFrame
    """
    for col in (col_a, col_b):
        if col not in dataframe.columns:
            raise DataValidationError(f"Column not found: '{col}'")
        if not pd.api.types.is_numeric_dtype(dataframe[col]):
            raise DataValidationError(f"Column '{col}' must be numeric for polynomial interaction")

    for name, power in (("power_a", power_a), ("power_b", power_b)):
        if power < 1:
            raise DataValidationError(f"{name} must be >= 1, got {power}")

    result = dataframe.copy()

    result[output_column] = (result[col_a] ** power_a) * (result[col_b] ** power_b)

    logger.info(
        "add_polynomial_interaction: created '%s' = %s^%d * %s^%d",
        output_column,
        col_a,
        power_a,
        col_b,
        power_b,
    )

    return result
