"""
Tests: preprocessing/scaler.py
Version: 0.6.0
"""

import pandas as pd
import pytest
from pipelineforge.exceptions import DataValidationError
from pipelineforge.preprocessing.scaler import scale_numeric_features

# ─────────────────────────────────────────────
# strategy: minmax
# ─────────────────────────────────────────────


def test_minmax_basic() -> None:
    df = pd.DataFrame({"a": [0.0, 5.0, 10.0]})
    result = scale_numeric_features(df, columns=["a"], strategy="minmax")
    assert result["a"].min() == pytest.approx(0.0)
    assert result["a"].max() == pytest.approx(1.0)
    assert result["a"].iloc[1] == pytest.approx(0.5)


def test_minmax_multiple_columns() -> None:
    df = pd.DataFrame({"a": [0.0, 10.0], "b": [5.0, 15.0]})
    result = scale_numeric_features(df, columns=["a", "b"], strategy="minmax")
    assert result["a"].min() == pytest.approx(0.0)
    assert result["b"].max() == pytest.approx(1.0)


def test_minmax_preserves_other_columns() -> None:
    df = pd.DataFrame({"a": [0.0, 10.0], "label": ["x", "y"]})
    result = scale_numeric_features(df, columns=["a"], strategy="minmax")
    assert result["label"].tolist() == ["x", "y"]


def test_minmax_constant_column_raises() -> None:
    df = pd.DataFrame({"a": [5.0, 5.0, 5.0]})
    with pytest.raises(DataValidationError, match="min == max"):
        scale_numeric_features(df, columns=["a"], strategy="minmax")


# ─────────────────────────────────────────────
# strategy: standard
# ─────────────────────────────────────────────


def test_standard_basic() -> None:
    df = pd.DataFrame({"a": [1.0, 2.0, 3.0]})
    result = scale_numeric_features(df, columns=["a"], strategy="standard")
    assert result["a"].mean() == pytest.approx(0.0, abs=1e-9)
    assert result["a"].std() == pytest.approx(1.0)


def test_standard_zero_std_raises() -> None:
    df = pd.DataFrame({"a": [3.0, 3.0, 3.0]})
    with pytest.raises(DataValidationError, match="std == 0"):
        scale_numeric_features(df, columns=["a"], strategy="standard")


# ─────────────────────────────────────────────
# edge cases
# ─────────────────────────────────────────────


def test_invalid_strategy_raises() -> None:
    df = pd.DataFrame({"a": [1.0, 2.0]})
    with pytest.raises(DataValidationError, match="Invalid strategy"):
        scale_numeric_features(df, columns=["a"], strategy="robust")


def test_missing_column_raises() -> None:
    df = pd.DataFrame({"a": [1.0, 2.0]})
    with pytest.raises(DataValidationError, match="not found"):
        scale_numeric_features(df, columns=["z"], strategy="minmax")


def test_non_numeric_column_raises() -> None:
    df = pd.DataFrame({"name": ["Alice", "Bob"]})
    with pytest.raises(DataValidationError, match="Non-numeric"):
        scale_numeric_features(df, columns=["name"], strategy="minmax")
