"""
Tests: features/generator.py
Version: 0.7.0
"""

import pandas as pd
import pytest
from pipelineforge.exceptions import DataValidationError
from pipelineforge.features.generator import (
    add_aggregate_feature,
    add_difference_feature,
    add_ratio_feature,
)

# ─────────────────────────────────────────────
# add_ratio_feature
# ─────────────────────────────────────────────


def test_ratio_basic() -> None:
    df = pd.DataFrame({"a": [10.0, 20.0], "b": [2.0, 4.0]})
    result = add_ratio_feature(df, "a", "b", "ratio")
    assert result["ratio"].tolist() == [5.0, 5.0]


def test_ratio_division_by_zero_fills() -> None:
    df = pd.DataFrame({"a": [10.0, 20.0], "b": [0.0, 4.0]})
    result = add_ratio_feature(df, "a", "b", "ratio", fill_value=-1.0)
    assert result["ratio"].iloc[0] == -1.0
    assert result["ratio"].iloc[1] == 5.0


def test_ratio_missing_column_raises() -> None:
    df = pd.DataFrame({"a": [1.0]})
    with pytest.raises(DataValidationError, match="not found"):
        add_ratio_feature(df, "a", "z", "ratio")


def test_ratio_non_numeric_raises() -> None:
    df = pd.DataFrame({"a": ["x", "y"], "b": [1.0, 2.0]})
    with pytest.raises(DataValidationError, match="numeric"):
        add_ratio_feature(df, "a", "b", "ratio")


def test_ratio_preserves_original_columns() -> None:
    df = pd.DataFrame({"a": [10.0], "b": [2.0]})
    result = add_ratio_feature(df, "a", "b", "ratio")
    assert "a" in result.columns
    assert "b" in result.columns


# ─────────────────────────────────────────────
# add_difference_feature
# ─────────────────────────────────────────────


def test_difference_basic() -> None:
    df = pd.DataFrame({"a": [10.0, 20.0], "b": [3.0, 5.0]})
    result = add_difference_feature(df, "a", "b", "diff")
    assert result["diff"].tolist() == [7.0, 15.0]


def test_difference_missing_column_raises() -> None:
    df = pd.DataFrame({"a": [1.0]})
    with pytest.raises(DataValidationError, match="not found"):
        add_difference_feature(df, "a", "z", "diff")


def test_difference_non_numeric_raises() -> None:
    df = pd.DataFrame({"a": ["x"], "b": [1.0]})
    with pytest.raises(DataValidationError, match="numeric"):
        add_difference_feature(df, "a", "b", "diff")


# ─────────────────────────────────────────────
# add_aggregate_feature
# ─────────────────────────────────────────────


def test_aggregate_sum() -> None:
    df = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]})
    result = add_aggregate_feature(df, ["a", "b"], "sum", "total")
    assert result["total"].tolist() == [4.0, 6.0]


def test_aggregate_mean() -> None:
    df = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]})
    result = add_aggregate_feature(df, ["a", "b"], "mean", "avg")
    assert result["avg"].tolist() == [2.0, 3.0]


def test_aggregate_min() -> None:
    df = pd.DataFrame({"a": [1.0, 5.0], "b": [3.0, 2.0]})
    result = add_aggregate_feature(df, ["a", "b"], "min", "minimum")
    assert result["minimum"].tolist() == [1.0, 2.0]


def test_aggregate_max() -> None:
    df = pd.DataFrame({"a": [1.0, 5.0], "b": [3.0, 2.0]})
    result = add_aggregate_feature(df, ["a", "b"], "max", "maximum")
    assert result["maximum"].tolist() == [3.0, 5.0]


def test_aggregate_invalid_operation_raises() -> None:
    df = pd.DataFrame({"a": [1.0], "b": [2.0]})
    with pytest.raises(DataValidationError, match="Invalid operation"):
        add_aggregate_feature(df, ["a", "b"], "median", "agg")


def test_aggregate_missing_column_raises() -> None:
    df = pd.DataFrame({"a": [1.0]})
    with pytest.raises(DataValidationError, match="not found"):
        add_aggregate_feature(df, ["a", "z"], "sum", "total")


def test_aggregate_non_numeric_raises() -> None:
    df = pd.DataFrame({"a": ["x"], "b": [1.0]})
    with pytest.raises(DataValidationError, match="Non-numeric"):
        add_aggregate_feature(df, ["a", "b"], "sum", "total")
