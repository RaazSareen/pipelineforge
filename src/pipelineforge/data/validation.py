"""
Module: validation.py
Project: PipelineForge
Version: 0.3.0
Last Updated: 2026-05-13

Purpose:
    Dataframe validation utilities.
"""

import pandas as pd

from pipelineforge.exceptions import DataValidationError


def validate_dataframe(
    dataframe: pd.DataFrame,
    required_columns: list[str] | None = None,
) -> bool:
    """
    Validate dataframe structure.

    Parameters
    ----------
    dataframe : pd.DataFrame
        Input dataframe.

    required_columns : list[str] | None
        Required dataframe columns.

    Returns
    -------
    bool
        Validation result.
    """

    if dataframe.empty:
        raise DataValidationError("Input dataframe is empty.")

    if required_columns:
        missing_columns = [column for column in required_columns if column not in dataframe.columns]

        if missing_columns:
            raise DataValidationError(
                f"Missing required columns: {missing_columns}",
            )

    return True
