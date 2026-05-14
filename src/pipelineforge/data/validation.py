"""
PipelineForge Module

Version:
    0.5.0

Updated:
    2026-05-14

Purpose:
    DataFrame validation utilities with granular validators and result reporting.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from pipelineforge.core.logging import setup_logger
from pipelineforge.exceptions import DataValidationError

logger = setup_logger(__name__)


# ─────────────────────────────────────────────
# ValidationResult
# ─────────────────────────────────────────────


@dataclass
class ValidationResult:
    """
    Collects validation outcomes without immediately raising.

    Use .raise_if_invalid() to enforce failures at a chosen point.
    """

    errors: list[str] = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        return len(self.errors) == 0

    def add_error(self, message: str) -> None:
        self.errors.append(message)
        logger.warning("Validation error: %s", message)

    def raise_if_invalid(self) -> None:
        """
        Raise DataValidationError if any errors were collected.
        """
        if not self.is_valid:
            raise DataValidationError(
                f"{len(self.errors)} validation error(s):\n"
                + "\n".join(f"  - {e}" for e in self.errors)
            )

    def __repr__(self) -> str:
        status = "PASS" if self.is_valid else f"FAIL ({len(self.errors)} error(s))"
        return f"ValidationResult(status={status})"


# ─────────────────────────────────────────────
# Granular Validators
# ─────────────────────────────────────────────


def validate_dataframe_not_empty(
    dataframe: pd.DataFrame,
    result: ValidationResult | None = None,
) -> ValidationResult:
    """
    Validate that the DataFrame is not empty.

    Parameters
    ----------
    dataframe : pd.DataFrame
        Input DataFrame.
    result : ValidationResult | None
        Existing result to append to. Creates new if None.

    Returns
    -------
    ValidationResult
    """
    result = result or ValidationResult()

    if dataframe.empty:
        result.add_error("DataFrame is empty.")

    return result


def validate_required_columns(
    dataframe: pd.DataFrame,
    required_columns: list[str],
    result: ValidationResult | None = None,
) -> ValidationResult:
    """
    Validate that all required columns are present.

    Parameters
    ----------
    dataframe : pd.DataFrame
        Input DataFrame.
    required_columns : list[str]
        Column names that must exist.
    result : ValidationResult | None
        Existing result to append to. Creates new if None.

    Returns
    -------
    ValidationResult
    """
    result = result or ValidationResult()

    missing = [col for col in required_columns if col not in dataframe.columns]

    if missing:
        result.add_error(f"Missing required columns: {missing}")

    return result


def validate_no_missing_values(
    dataframe: pd.DataFrame,
    columns: list[str] | None = None,
    result: ValidationResult | None = None,
) -> ValidationResult:
    """
    Validate that specified columns (or all columns) have no missing values.

    Parameters
    ----------
    dataframe : pd.DataFrame
        Input DataFrame.
    columns : list[str] | None
        Columns to check. Checks all columns if None.
    result : ValidationResult | None
        Existing result to append to. Creates new if None.

    Returns
    -------
    ValidationResult
    """
    result = result or ValidationResult()

    target_cols = columns if columns is not None else dataframe.columns.tolist()

    for col in target_cols:
        if col not in dataframe.columns:
            result.add_error(f"Column not found for null check: '{col}'")
            continue

        null_count = dataframe[col].isnull().sum()

        if null_count > 0:
            result.add_error(f"Column '{col}' has {null_count} missing value(s).")

    return result


def validate_column_types(
    dataframe: pd.DataFrame,
    expected_types: dict[str, type],
    result: ValidationResult | None = None,
) -> ValidationResult:
    """
    Validate that columns match expected Python/numpy dtypes.

    Parameters
    ----------
    dataframe : pd.DataFrame
        Input DataFrame.
    expected_types : dict[str, type]
        Mapping of column name to expected dtype.
        Example: {"age": int, "score": float, "name": str}
    result : ValidationResult | None
        Existing result to append to. Creates new if None.

    Returns
    -------
    ValidationResult
    """
    result = result or ValidationResult()

    # numpy dtype.kind: 'i'=signed int, 'u'=unsigned int, 'f'=float, 'O'=object/str, 'b'=bool
    _DTYPE_KIND_MAP: dict[type, set[str]] = {
        int: {"i", "u"},
        float: {"f"},
        str: {"O"},
        bool: {"b"},
    }

    for col, expected in expected_types.items():
        if col not in dataframe.columns:
            result.add_error(f"Column not found for type check: '{col}'")
            continue

        actual_kind = dataframe[col].dtype.kind
        expected_kinds = _DTYPE_KIND_MAP.get(expected)

        if expected_kinds and actual_kind not in expected_kinds:
            result.add_error(
                f"Column '{col}' expected type '{expected.__name__}', "
                f"got dtype '{dataframe[col].dtype}'."
            )

    return result


def check_duplicate_rows(
    dataframe: pd.DataFrame,
    subset: list[str] | None = None,
    result: ValidationResult | None = None,
) -> ValidationResult:
    """
    Check for duplicate rows in the DataFrame.

    Parameters
    ----------
    dataframe : pd.DataFrame
        Input DataFrame.
    subset : list[str] | None
        Columns to consider for duplicate detection. All columns if None.
    result : ValidationResult | None
        Existing result to append to. Creates new if None.

    Returns
    -------
    ValidationResult
    """
    result = result or ValidationResult()

    duplicate_count = dataframe.duplicated(subset=subset).sum()

    if duplicate_count > 0:
        scope = f"columns {subset}" if subset else "all columns"
        result.add_error(f"Found {duplicate_count} duplicate row(s) across {scope}.")

    return result


# ─────────────────────────────────────────────
# Composite Validator
# ─────────────────────────────────────────────


def validate_dataframe(
    dataframe: pd.DataFrame,
    required_columns: list[str] | None = None,
) -> bool:
    """
    Backward-compatible composite validator.

    Checks: not empty + required columns present.
    Raises DataValidationError on first failure group.

    Parameters
    ----------
    dataframe : pd.DataFrame
        Input DataFrame.
    required_columns : list[str] | None
        Required column names.

    Returns
    -------
    bool
        True if all checks pass.
    """
    result = ValidationResult()

    validate_dataframe_not_empty(dataframe, result=result)

    if required_columns:
        validate_required_columns(dataframe, required_columns, result=result)

    result.raise_if_invalid()

    return True
