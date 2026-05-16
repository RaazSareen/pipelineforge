"""
Tests: preprocessing/imputer.py
Version: 0.6.0
"""

import pandas as pd
import pytest
from pipelineforge.exceptions import DataValidationError
from pipelineforge.preprocessing.imputer import fill_missing_values

# ─────────────────────────────────────────────
# strategy: mean
# ─────────────────────────────────────────────


def test_fill_mean_basic() -> None:
    df = pd.DataFrame({"a": [1.0, 2.0, None]})
    result = fill_missing_values(df, strategy="mean", columns=["a"])
    assert result["a"].isnull().sum() == 0
    assert result["a"].iloc[2] == pytest.approx(1.5)


def test_fill_mean_non_numeric_raises() -> None:
    df = pd.DataFrame({"name": ["Alice", None]})
    with pytest.raises(DataValidationError, match="numeric"):
        fill_missing_values(df, strategy="mean", columns=["name"])


# ─────────────────────────────────────────────
# strategy: median
# ─────────────────────────────────────────────


def test_fill_median_basic() -> None:
    df = pd.DataFrame({"a": [1.0, 3.0, None]})
    result = fill_missing_values(df, strategy="median", columns=["a"])
    assert result["a"].isnull().sum() == 0
    assert result["a"].iloc[2] == pytest.approx(2.0)


def test_fill_median_non_numeric_raises() -> None:
    df = pd.DataFrame({"name": ["Alice", None]})
    with pytest.raises(DataValidationError, match="numeric"):
        fill_missing_values(df, strategy="median", columns=["name"])


# ─────────────────────────────────────────────
# strategy: mode
# ─────────────────────────────────────────────


def test_fill_mode_numeric() -> None:
    df = pd.DataFrame({"a": [1, 1, 2, None]})
    result = fill_missing_values(df, strategy="mode", columns=["a"])
    assert result["a"].isnull().sum() == 0
    assert result["a"].iloc[3] == 1


def test_fill_mode_string() -> None:
    df = pd.DataFrame({"cat": ["A", "A", "B", None]})
    result = fill_missing_values(df, strategy="mode", columns=["cat"])
    assert result["cat"].isnull().sum() == 0
    assert result["cat"].iloc[3] == "A"


# ─────────────────────────────────────────────
# strategy: constant
# ─────────────────────────────────────────────


def test_fill_constant_numeric() -> None:
    df = pd.DataFrame({"a": [1.0, None, 3.0]})
    result = fill_missing_values(df, strategy="constant", columns=["a"], fill_value=0.0)
    assert result["a"].iloc[1] == 0.0


def test_fill_constant_string() -> None:
    df = pd.DataFrame({"label": ["X", None]})
    result = fill_missing_values(df, strategy="constant", columns=["label"], fill_value="unknown")
    assert result["label"].iloc[1] == "unknown"


def test_fill_constant_missing_fill_value_raises() -> None:
    df = pd.DataFrame({"a": [1.0, None]})
    with pytest.raises(DataValidationError, match="fill_value"):
        fill_missing_values(df, strategy="constant", columns=["a"])


# ─────────────────────────────────────────────
# edge cases
# ─────────────────────────────────────────────


def test_fill_invalid_strategy_raises() -> None:
    df = pd.DataFrame({"a": [1.0, None]})
    with pytest.raises(DataValidationError, match="Invalid strategy"):
        fill_missing_values(df, strategy="interpolate", columns=["a"])


def test_fill_missing_column_raises() -> None:
    df = pd.DataFrame({"a": [1.0]})
    with pytest.raises(DataValidationError, match="not found"):
        fill_missing_values(df, strategy="mean", columns=["z"])


def test_fill_no_nulls_no_change() -> None:
    df = pd.DataFrame({"a": [1.0, 2.0, 3.0]})
    result = fill_missing_values(df, strategy="mean", columns=["a"])
    assert result["a"].tolist() == [1.0, 2.0, 3.0]


def test_fill_all_columns_auto() -> None:
    df = pd.DataFrame({"a": [1.0, None], "b": [None, 2.0]})
    result = fill_missing_values(df, strategy="mean")
    assert result["a"].isnull().sum() == 0
    assert result["b"].isnull().sum() == 0
