"""
Tests: preprocessing/cleaner.py
Version: 0.6.0
"""

import pandas as pd
import pytest
from pipelineforge.exceptions import DataValidationError
from pipelineforge.preprocessing.cleaner import (
    cast_column_types,
    drop_duplicate_rows,
    drop_missing_rows,
    rename_columns,
    strip_whitespace,
)

# ─────────────────────────────────────────────
# drop_missing_rows
# ─────────────────────────────────────────────


def test_drop_missing_rows_removes_nulls() -> None:
    df = pd.DataFrame({"a": [1, None, 3], "b": [4, 5, 6]})
    result = drop_missing_rows(df)
    assert len(result) == 2
    assert result["a"].isnull().sum() == 0


def test_drop_missing_rows_subset() -> None:
    df = pd.DataFrame({"a": [1, None, 3], "b": [None, None, 6]})
    result = drop_missing_rows(df, subset=["a"])
    assert len(result) == 2


def test_drop_missing_rows_threshold() -> None:
    df = pd.DataFrame({"a": [1, None, None], "b": [2, None, 6]})
    # Row 1: 0/2 null = 0.0 → keep
    # Row 2: 2/2 null = 1.0 → drop (> 0.5)
    # Row 3: 1/2 null = 0.5 → keep (= threshold, not exceeding)
    result = drop_missing_rows(df, threshold=0.5)
    assert len(result) == 2


def test_drop_missing_rows_invalid_threshold() -> None:
    df = pd.DataFrame({"a": [1, 2]})
    with pytest.raises(DataValidationError, match="threshold"):
        drop_missing_rows(df, threshold=1.5)


def test_drop_missing_rows_no_change() -> None:
    df = pd.DataFrame({"a": [1, 2, 3]})
    result = drop_missing_rows(df)
    assert len(result) == 3


# ─────────────────────────────────────────────
# drop_duplicate_rows
# ─────────────────────────────────────────────


def test_drop_duplicate_rows_removes_dupes() -> None:
    df = pd.DataFrame({"a": [1, 1, 2]})
    result = drop_duplicate_rows(df)
    assert len(result) == 2


def test_drop_duplicate_rows_subset() -> None:
    df = pd.DataFrame({"a": [1, 1], "b": [1, 2]})
    result = drop_duplicate_rows(df, subset=["a"])
    assert len(result) == 1


def test_drop_duplicate_rows_no_dupes() -> None:
    df = pd.DataFrame({"a": [1, 2, 3]})
    result = drop_duplicate_rows(df)
    assert len(result) == 3


# ─────────────────────────────────────────────
# strip_whitespace
# ─────────────────────────────────────────────


def test_strip_whitespace_removes_padding() -> None:
    df = pd.DataFrame({"name": ["  Alice  ", " Bob "]})
    result = strip_whitespace(df)
    assert result["name"].tolist() == ["Alice", "Bob"]


def test_strip_whitespace_specific_columns() -> None:
    df = pd.DataFrame({"a": ["  x  "], "b": ["  y  "]})
    result = strip_whitespace(df, columns=["a"])
    assert result["a"][0] == "x"
    assert result["b"][0] == "  y  "


def test_strip_whitespace_skips_missing_column(caplog) -> None:
    df = pd.DataFrame({"a": ["  x  "]})
    result = strip_whitespace(df, columns=["z"])
    assert "z" not in result.columns


# ─────────────────────────────────────────────
# rename_columns
# ─────────────────────────────────────────────


def test_rename_columns_success() -> None:
    df = pd.DataFrame({"old_name": [1, 2]})
    result = rename_columns(df, {"old_name": "new_name"})
    assert "new_name" in result.columns
    assert "old_name" not in result.columns


def test_rename_columns_missing_column() -> None:
    df = pd.DataFrame({"a": [1]})
    with pytest.raises(DataValidationError, match="not found"):
        rename_columns(df, {"z": "new_z"})


# ─────────────────────────────────────────────
# cast_column_types
# ─────────────────────────────────────────────


def test_cast_column_types_int_to_float() -> None:
    df = pd.DataFrame({"age": [25, 30]})
    result = cast_column_types(df, {"age": float})
    assert result["age"].dtype == float


def test_cast_column_types_str_to_int() -> None:
    df = pd.DataFrame({"count": ["1", "2", "3"]})
    result = cast_column_types(df, {"count": int})
    assert result["count"].dtype == int


def test_cast_column_types_invalid_cast() -> None:
    df = pd.DataFrame({"name": ["alice", "bob"]})
    with pytest.raises(DataValidationError, match="Cannot cast"):
        cast_column_types(df, {"name": int})


def test_cast_column_types_missing_column() -> None:
    df = pd.DataFrame({"a": [1]})
    with pytest.raises(DataValidationError, match="not found"):
        cast_column_types(df, {"z": float})
