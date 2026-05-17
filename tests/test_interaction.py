"""
Tests: features/interaction.py
Version: 0.7.0
"""

import pandas as pd
import pytest
from pipelineforge.exceptions import DataValidationError
from pipelineforge.features.interaction import (
    add_division_interaction,
    add_interaction_features,
    add_polynomial_interaction,
)

# ─────────────────────────────────────────────
# add_interaction_features
# ─────────────────────────────────────────────


def test_interaction_basic() -> None:
    df = pd.DataFrame({"a": [2.0, 3.0], "b": [4.0, 5.0]})
    result = add_interaction_features(df, columns=["a", "b"])
    assert "a_x_b" in result.columns
    assert result["a_x_b"].tolist() == [8.0, 15.0]


def test_interaction_three_columns() -> None:
    df = pd.DataFrame({"a": [1.0], "b": [2.0], "c": [3.0]})
    result = add_interaction_features(df, columns=["a", "b", "c"])
    assert "a_x_b" in result.columns
    assert "a_x_c" in result.columns
    assert "b_x_c" in result.columns


def test_interaction_custom_separator() -> None:
    df = pd.DataFrame({"a": [1.0], "b": [2.0]})
    result = add_interaction_features(df, columns=["a", "b"], separator="_mul_")
    assert "a_mul_b" in result.columns


def test_interaction_preserves_originals() -> None:
    df = pd.DataFrame({"a": [2.0], "b": [3.0]})
    result = add_interaction_features(df, columns=["a", "b"])
    assert "a" in result.columns
    assert "b" in result.columns


def test_interaction_too_few_columns_raises() -> None:
    df = pd.DataFrame({"a": [1.0]})
    with pytest.raises(DataValidationError, match="At least 2"):
        add_interaction_features(df, columns=["a"])


def test_interaction_missing_column_raises() -> None:
    df = pd.DataFrame({"a": [1.0]})
    with pytest.raises(DataValidationError, match="not found"):
        add_interaction_features(df, columns=["a", "z"])


def test_interaction_non_numeric_raises() -> None:
    df = pd.DataFrame({"a": ["x", "y"], "b": [1.0, 2.0]})
    with pytest.raises(DataValidationError, match="Non-numeric"):
        add_interaction_features(df, columns=["a", "b"])


# ─────────────────────────────────────────────
# add_division_interaction
# ─────────────────────────────────────────────


def test_division_interaction_basic() -> None:
    df = pd.DataFrame({"a": [10.0, 20.0], "b": [2.0, 4.0]})
    result = add_division_interaction(df, "a", "b", "a_div_b")
    assert result["a_div_b"].tolist() == [5.0, 5.0]


def test_division_interaction_zero_denominator() -> None:
    df = pd.DataFrame({"a": [10.0], "b": [0.0]})
    result = add_division_interaction(df, "a", "b", "out", fill_value=-999.0)
    assert result["out"].iloc[0] == -999.0


def test_division_interaction_missing_column_raises() -> None:
    df = pd.DataFrame({"a": [1.0]})
    with pytest.raises(DataValidationError, match="not found"):
        add_division_interaction(df, "a", "z", "out")


def test_division_interaction_non_numeric_raises() -> None:
    df = pd.DataFrame({"a": ["x"], "b": [1.0]})
    with pytest.raises(DataValidationError, match="numeric"):
        add_division_interaction(df, "a", "b", "out")


# ─────────────────────────────────────────────
# add_polynomial_interaction
# ─────────────────────────────────────────────


def test_polynomial_interaction_basic() -> None:
    df = pd.DataFrame({"a": [2.0, 3.0], "b": [3.0, 4.0]})
    result = add_polynomial_interaction(df, "a", "b", "out", power_a=2, power_b=1)
    assert result["out"].tolist() == [12.0, 36.0]


def test_polynomial_interaction_default_powers() -> None:
    df = pd.DataFrame({"a": [2.0], "b": [5.0]})
    result = add_polynomial_interaction(df, "a", "b", "out")
    assert result["out"].iloc[0] == 10.0


def test_polynomial_interaction_invalid_power_raises() -> None:
    df = pd.DataFrame({"a": [1.0], "b": [2.0]})
    with pytest.raises(DataValidationError, match="power_a"):
        add_polynomial_interaction(df, "a", "b", "out", power_a=0)


def test_polynomial_interaction_missing_column_raises() -> None:
    df = pd.DataFrame({"a": [1.0]})
    with pytest.raises(DataValidationError, match="not found"):
        add_polynomial_interaction(df, "a", "z", "out")


def test_polynomial_interaction_non_numeric_raises() -> None:
    df = pd.DataFrame({"a": ["x"], "b": [1.0]})
    with pytest.raises(DataValidationError, match="numeric"):
        add_polynomial_interaction(df, "a", "b", "out")
