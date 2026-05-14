"""
Tests: validation.py
"""

import pandas as pd
import pytest
from pipelineforge.data.validation import (
    ValidationResult,
    check_duplicate_rows,
    validate_column_types,
    validate_dataframe,
    validate_dataframe_not_empty,
    validate_no_missing_values,
    validate_required_columns,
)
from pipelineforge.exceptions import DataValidationError

# ─────────────────────────────────────────────
# ValidationResult
# ─────────────────────────────────────────────


def test_validation_result_pass() -> None:
    result = ValidationResult()
    assert result.is_valid is True


def test_validation_result_fail() -> None:
    result = ValidationResult()
    result.add_error("something wrong")
    assert result.is_valid is False
    assert len(result.errors) == 1


def test_validation_result_raise_if_invalid() -> None:
    result = ValidationResult()
    result.add_error("col missing")

    with pytest.raises(DataValidationError, match="1 validation error"):
        result.raise_if_invalid()


def test_validation_result_no_raise_when_valid() -> None:
    result = ValidationResult()
    result.raise_if_invalid()


# ─────────────────────────────────────────────
# validate_dataframe_not_empty
# ─────────────────────────────────────────────


def test_not_empty_pass() -> None:
    df = pd.DataFrame({"a": [1, 2]})
    result = validate_dataframe_not_empty(df)
    assert result.is_valid


def test_not_empty_fail() -> None:
    df = pd.DataFrame()
    result = validate_dataframe_not_empty(df)
    assert not result.is_valid
    assert any("empty" in e for e in result.errors)


# ─────────────────────────────────────────────
# validate_required_columns
# ─────────────────────────────────────────────


def test_required_columns_pass() -> None:
    df = pd.DataFrame({"feature": [1], "target": [0]})
    result = validate_required_columns(df, ["feature", "target"])
    assert result.is_valid


def test_required_columns_fail() -> None:
    df = pd.DataFrame({"feature": [1]})
    result = validate_required_columns(df, ["feature", "target"])
    assert not result.is_valid
    assert any("target" in e for e in result.errors)


# ─────────────────────────────────────────────
# validate_no_missing_values
# ─────────────────────────────────────────────


def test_no_missing_values_pass() -> None:
    df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    result = validate_no_missing_values(df)
    assert result.is_valid


def test_no_missing_values_fail() -> None:
    df = pd.DataFrame({"a": [1, None, 3]})
    result = validate_no_missing_values(df, columns=["a"])
    assert not result.is_valid
    assert any("missing" in e for e in result.errors)


def test_no_missing_values_unknown_column() -> None:
    df = pd.DataFrame({"a": [1, 2]})
    result = validate_no_missing_values(df, columns=["z"])
    assert not result.is_valid
    assert any("not found" in e for e in result.errors)


# ─────────────────────────────────────────────
# validate_column_types
# ─────────────────────────────────────────────


def test_column_types_pass() -> None:
    df = pd.DataFrame({"age": [25, 30], "score": [0.9, 0.8]})
    result = validate_column_types(df, {"age": int, "score": float})
    assert result.is_valid


def test_column_types_fail() -> None:
    df = pd.DataFrame({"age": ["twenty", "thirty"]})
    result = validate_column_types(df, {"age": int})
    assert not result.is_valid
    assert any("age" in e for e in result.errors)


def test_column_types_missing_column() -> None:
    df = pd.DataFrame({"a": [1]})
    result = validate_column_types(df, {"z": int})
    assert not result.is_valid
    assert any("not found" in e for e in result.errors)


# ─────────────────────────────────────────────
# check_duplicate_rows
# ─────────────────────────────────────────────


def test_duplicate_rows_pass() -> None:
    df = pd.DataFrame({"a": [1, 2, 3]})
    result = check_duplicate_rows(df)
    assert result.is_valid


def test_duplicate_rows_fail() -> None:
    df = pd.DataFrame({"a": [1, 1, 2]})
    result = check_duplicate_rows(df)
    assert not result.is_valid
    assert any("duplicate" in e for e in result.errors)


def test_duplicate_rows_subset_pass() -> None:
    df = pd.DataFrame({"a": [1, 1], "b": [1, 2]})
    result = check_duplicate_rows(df, subset=["a", "b"])
    assert result.is_valid


# ─────────────────────────────────────────────
# validate_dataframe (backward-compatible)
# ─────────────────────────────────────────────


def test_validate_dataframe_success() -> None:
    df = pd.DataFrame({"feature": [1, 2, 3]})
    assert validate_dataframe(df, required_columns=["feature"]) is True


def test_validate_dataframe_empty() -> None:
    df = pd.DataFrame()
    with pytest.raises(DataValidationError):
        validate_dataframe(df)


def test_validate_dataframe_missing_column() -> None:
    df = pd.DataFrame({"feature": [1, 2, 3]})
    with pytest.raises(DataValidationError):
        validate_dataframe(df, required_columns=["target"])


# ─────────────────────────────────────────────
# Chaining — shared result accumulation
# ─────────────────────────────────────────────


def test_chained_validators_accumulate_errors() -> None:
    df = pd.DataFrame({"a": [1, None]})
    result = ValidationResult()

    validate_required_columns(df, ["a", "b"], result=result)
    validate_no_missing_values(df, columns=["a"], result=result)

    assert not result.is_valid
    assert len(result.errors) == 2
