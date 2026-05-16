"""
PipelineForge Module

Version:
    0.6.0

Updated:
    2026-05-16

Purpose:
    Categorical encoding utilities.
"""

from __future__ import annotations

import pandas as pd

from pipelineforge.core.logging import setup_logger
from pipelineforge.exceptions import DataValidationError

logger = setup_logger(__name__)


def encode_categorical(
    dataframe: pd.DataFrame,
    columns: list[str],
    strategy: str = "onehot",
    drop_original: bool = True,
) -> pd.DataFrame:
    """
    Encode categorical columns.

    Parameters
    ----------
    dataframe : pd.DataFrame
    columns : list[str]
        Columns to encode.
    strategy : str
        One of: 'onehot', 'label'.
    drop_original : bool
        If True, drop original columns after one-hot encoding.

    Returns
    -------
    pd.DataFrame
    """
    valid_strategies = {"onehot", "label"}

    if strategy not in valid_strategies:
        raise DataValidationError(
            f"Invalid strategy: '{strategy}'. Must be one of {sorted(valid_strategies)}"
        )

    missing = [col for col in columns if col not in dataframe.columns]
    if missing:
        raise DataValidationError(f"Columns not found for encoding: {missing}")

    result = dataframe.copy()

    if strategy == "onehot":
        result = pd.get_dummies(result, columns=columns, drop_first=False)
        if not drop_original:
            for col in columns:
                result[col] = dataframe[col]
        logger.info("encode_categorical: one-hot encoded %d column(s)", len(columns))

    elif strategy == "label":
        for col in columns:
            categories = result[col].dropna().unique()
            label_map = {val: idx for idx, val in enumerate(sorted(categories))}
            result[col] = result[col].map(label_map)
            logger.info("encode_categorical: label encoded '%s' → %s", col, label_map)

    return result
