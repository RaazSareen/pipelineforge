"""
Tests: preprocessing/selector.py
Version: 0.6.0
"""

import pandas as pd
import pytest
from pipelineforge.exceptions import DataValidationError
from pipelineforge.preprocessing.selector import (
    drop_columns,
    drop_low_variance,
    select_columns,
)

# ─────────────────────────────────────────────
# drop_columns
# ─────────────────────────────────────────────


def test_drop_columns_basic() -> None:
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4], "c": [5, 6]})
    result = drop_columns(df, columns=["b", "c"])
    assert list(result.columns) == ["a"]


def test_drop_columns_single() -> None:
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    result = drop_columns(df, columns=["b"])
    assert "b" not in result.columns
    assert "a" in result.columns


def test_drop_columns_missing_raises() -> None:
    df = pd.DataFrame({"a": [1, 2]})
    with pytest.raises(DataValidationError, match="not found"):
        drop_columns(df, columns=["z"])


# ─────────────────────────────────────────────
# select_columns
# ─────────────────────────────────────────────


def test_select_columns_basic() -> None:
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4], "c": [5, 6]})
    result = select_columns(df, columns=["a", "c"])
    assert list(result.columns) == ["a", "c"]
    assert "b" not in result.columns


def test_select_columns_preserves_values() -> None:
    df = pd.DataFrame({"a": [10, 20], "b": [30, 40]})
    result = select_columns(df, columns=["a"])
    assert result["a"].tolist() == [10, 20]


def test_select_columns_missing_raises() -> None:
    df = pd.DataFrame({"a": [1, 2]})
    with pytest.raises(DataValidationError, match="not found"):
        select_columns(df, columns=["z"])


# ─────────────────────────────────────────────
# drop_low_variance
# ─────────────────────────────────────────────


def test_drop_low_variance_removes_constant() -> None:
    df = pd.DataFrame({"a": [1, 1, 1], "b": [1, 2, 3]})
    result = drop_low_variance(df)
    assert "a" not in result.columns
    assert "b" in result.columns


def test_drop_low_variance_custom_threshold() -> None:
    df = pd.DataFrame({"a": [1, 2, 3], "b": [100, 200, 300]})
    result = drop_low_variance(df, threshold=1.5)
    assert "a" not in result.columns
    assert "b" in result.columns


def test_drop_low_variance_specific_columns() -> None:
    df = pd.DataFrame({"a": [1, 1, 1], "b": [1, 2, 3]})
    result = drop_low_variance(df, columns=["b"])
    assert "a" in result.columns
    assert "b" in result.columns


def test_drop_low_variance_negative_threshold_raises() -> None:
    df = pd.DataFrame({"a": [1, 2, 3]})
    with pytest.raises(DataValidationError, match="threshold"):
        drop_low_variance(df, threshold=-1.0)


def test_drop_low_variance_missing_column_raises() -> None:
    df = pd.DataFrame({"a": [1, 2, 3]})
    with pytest.raises(DataValidationError, match="not found"):
        drop_low_variance(df, columns=["z"])
