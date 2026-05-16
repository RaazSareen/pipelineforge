"""
PipelineForge Module

Version:
    0.6.0

Updated:
    2026-05-14

Purpose:
    DataFrame cleaning utilities.
"""

from __future__ import annotations

import pandas as pd

from pipelineforge.core.logging import setup_logger
from pipelineforge.exceptions import DataValidationError

logger = setup_logger(__name__)


def drop_missing_rows(
    dataframe: pd.DataFrame,
    subset: list[str] | None = None,
    threshold: float | None = None,
) -> pd.DataFrame:
    """
    Drop rows with missing values.

    Parameters
    ----------
    dataframe : pd.DataFrame
    subset : list[str] | None
        Limit check to these columns. All columns if None.
    threshold : float | None
        If set (0.0–1.0), drop rows where fraction of nulls exceeds threshold.
        If None, drop any row with at least one null.

    Returns
    -------
    pd.DataFrame
    """
    if threshold is not None:
        if not (0.0 <= threshold <= 1.0):
            raise DataValidationError(f"threshold must be between 0.0 and 1.0, got {threshold}")
        cols = subset if subset is not None else dataframe.columns.tolist()
        null_fraction = dataframe[cols].isnull().mean(axis=1)
        mask = null_fraction <= threshold
        dropped = (~mask).sum()
        result = dataframe[mask].reset_index(drop=True)
    else:
        before = len(dataframe)
        result = dataframe.dropna(subset=subset).reset_index(drop=True)
        dropped = before - len(result)

    if dropped > 0:
        logger.info("drop_missing_rows: dropped %d row(s)", dropped)

    return result


def drop_duplicate_rows(
    dataframe: pd.DataFrame,
    subset: list[str] | None = None,
    keep: str = "first",
) -> pd.DataFrame:
    """
    Drop duplicate rows.

    Parameters
    ----------
    dataframe : pd.DataFrame
    subset : list[str] | None
        Columns to consider. All columns if None.
    keep : str
        Which duplicate to keep: 'first', 'last', or False.

    Returns
    -------
    pd.DataFrame
    """
    before = len(dataframe)
    result = dataframe.drop_duplicates(subset=subset, keep=keep).reset_index(drop=True)
    dropped = before - len(result)

    if dropped > 0:
        logger.info("drop_duplicate_rows: dropped %d duplicate(s)", dropped)

    return result


def strip_whitespace(
    dataframe: pd.DataFrame,
    columns: list[str] | None = None,
) -> pd.DataFrame:
    """
    Strip leading/trailing whitespace from string columns.

    Parameters
    ----------
    dataframe : pd.DataFrame
    columns : list[str] | None
        Columns to process. Auto-detects string/object dtype columns if None.

    Returns
    -------
    pd.DataFrame
    """
    result = dataframe.copy()

    if columns is not None:
        target_cols = columns
    else:
        target_cols = [
            col
            for col in result.columns
            if pd.api.types.is_string_dtype(result[col])
            or pd.api.types.is_object_dtype(result[col])
        ]

    for col in target_cols:
        if col not in result.columns:
            logger.warning("strip_whitespace: column '%s' not found — skipped", col)
            continue
        if pd.api.types.is_string_dtype(result[col]) or pd.api.types.is_object_dtype(result[col]):
            result[col] = result[col].str.strip()

    logger.info("strip_whitespace: processed %d column(s)", len(target_cols))

    return result


def rename_columns(
    dataframe: pd.DataFrame,
    mapping: dict[str, str],
) -> pd.DataFrame:
    """
    Rename columns via a mapping dict.

    Parameters
    ----------
    dataframe : pd.DataFrame
    mapping : dict[str, str]
        Old name → new name.

    Returns
    -------
    pd.DataFrame
    """
    missing = [col for col in mapping if col not in dataframe.columns]

    if missing:
        raise DataValidationError(f"Columns not found for rename: {missing}")

    result = dataframe.rename(columns=mapping)

    logger.info("rename_columns: renamed %d column(s)", len(mapping))

    return result


def cast_column_types(
    dataframe: pd.DataFrame,
    type_map: dict[str, type],
) -> pd.DataFrame:
    """
    Cast columns to specified Python types.

    Parameters
    ----------
    dataframe : pd.DataFrame
    type_map : dict[str, type]
        Column name → target Python type (int, float, str, bool).

    Returns
    -------
    pd.DataFrame
    """
    result = dataframe.copy()

    for col, dtype in type_map.items():
        if col not in result.columns:
            raise DataValidationError(f"Column not found for cast: '{col}'")
        try:
            result[col] = result[col].astype(dtype)
        except (ValueError, TypeError) as exc:
            raise DataValidationError(
                f"Cannot cast column '{col}' to {dtype.__name__}: {exc}"
            ) from exc

    logger.info("cast_column_types: cast %d column(s)", len(type_map))

    return result
