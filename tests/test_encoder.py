"""
Tests: preprocessing/encoder.py
Version: 0.6.0
"""

import pandas as pd
import pytest
from pipelineforge.exceptions import DataValidationError
from pipelineforge.preprocessing.encoder import encode_categorical

# ─────────────────────────────────────────────
# strategy: onehot
# ─────────────────────────────────────────────


def test_onehot_basic() -> None:
    df = pd.DataFrame({"color": ["red", "blue", "red"]})
    result = encode_categorical(df, columns=["color"], strategy="onehot")
    assert "color_red" in result.columns
    assert "color_blue" in result.columns
    assert "color" not in result.columns


def test_onehot_multiple_columns() -> None:
    df = pd.DataFrame({"color": ["red", "blue"], "size": ["S", "M"]})
    result = encode_categorical(df, columns=["color", "size"], strategy="onehot")
    assert "color_red" in result.columns
    assert "size_S" in result.columns
    assert "color" not in result.columns
    assert "size" not in result.columns


def test_onehot_preserves_other_columns() -> None:
    df = pd.DataFrame({"color": ["red", "blue"], "value": [10, 20]})
    result = encode_categorical(df, columns=["color"], strategy="onehot")
    assert "value" in result.columns
    assert result["value"].tolist() == [10, 20]


# ─────────────────────────────────────────────
# strategy: label
# ─────────────────────────────────────────────


def test_label_basic() -> None:
    df = pd.DataFrame({"color": ["red", "blue", "green"]})
    result = encode_categorical(df, columns=["color"], strategy="label")
    assert result["color"].dtype in [int, "int64", "int32"]
    assert set(result["color"].tolist()) == {0, 1, 2}


def test_label_preserves_other_columns() -> None:
    df = pd.DataFrame({"color": ["red", "blue"], "value": [10, 20]})
    result = encode_categorical(df, columns=["color"], strategy="label")
    assert "value" in result.columns
    assert result["value"].tolist() == [10, 20]


def test_label_sorted_mapping() -> None:
    df = pd.DataFrame({"grade": ["C", "A", "B"]})
    result = encode_categorical(df, columns=["grade"], strategy="label")
    # A→0, B→1, C→2
    assert result["grade"].tolist() == [2, 0, 1]


# ─────────────────────────────────────────────
# edge cases
# ─────────────────────────────────────────────


def test_invalid_strategy_raises() -> None:
    df = pd.DataFrame({"color": ["red", "blue"]})
    with pytest.raises(DataValidationError, match="Invalid strategy"):
        encode_categorical(df, columns=["color"], strategy="binary")


def test_missing_column_raises() -> None:
    df = pd.DataFrame({"color": ["red", "blue"]})
    with pytest.raises(DataValidationError, match="not found"):
        encode_categorical(df, columns=["size"], strategy="onehot")
