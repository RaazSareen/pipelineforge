"""
Tests: features/transformer.py
Version: 0.7.0
"""

import pandas as pd
import pytest
from pipelineforge.exceptions import DataValidationError
from pipelineforge.features.transformer import (
    apply_binning,
    apply_log_transform,
    apply_polynomial_features,
    extract_datetime_features,
)

# ─────────────────────────────────────────────
# apply_log_transform
# ─────────────────────────────────────────────


def test_log_transform_basic() -> None:
    df = pd.DataFrame({"a": [0.0, 1.0, 3.0]})
    result = apply_log_transform(df, columns=["a"])
    assert "a_log" in result.columns
    assert result["a_log"].iloc[0] == pytest.approx(0.0)


def test_log_transform_preserves_original() -> None:
    df = pd.DataFrame({"a": [1.0, 2.0]})
    result = apply_log_transform(df, columns=["a"])
    assert "a" in result.columns


def test_log_transform_custom_suffix() -> None:
    df = pd.DataFrame({"a": [1.0, 2.0]})
    result = apply_log_transform(df, columns=["a"], output_suffix="_transformed")
    assert "a_transformed" in result.columns


def test_log_transform_negative_raises() -> None:
    df = pd.DataFrame({"a": [-1.0, 2.0]})
    with pytest.raises(DataValidationError, match="negative"):
        apply_log_transform(df, columns=["a"])


def test_log_transform_non_numeric_raises() -> None:
    df = pd.DataFrame({"a": ["x", "y"]})
    with pytest.raises(DataValidationError, match="Non-numeric"):
        apply_log_transform(df, columns=["a"])


def test_log_transform_missing_column_raises() -> None:
    df = pd.DataFrame({"a": [1.0]})
    with pytest.raises(DataValidationError, match="not found"):
        apply_log_transform(df, columns=["z"])


# ─────────────────────────────────────────────
# apply_polynomial_features
# ─────────────────────────────────────────────


def test_polynomial_degree2() -> None:
    df = pd.DataFrame({"a": [2.0, 3.0]})
    result = apply_polynomial_features(df, columns=["a"], degree=2)
    assert "a_pow2" in result.columns
    assert result["a_pow2"].tolist() == [4.0, 9.0]


def test_polynomial_degree3() -> None:
    df = pd.DataFrame({"a": [2.0, 3.0]})
    result = apply_polynomial_features(df, columns=["a"], degree=3)
    assert "a_pow2" in result.columns
    assert "a_pow3" in result.columns
    assert result["a_pow3"].tolist() == [8.0, 27.0]


def test_polynomial_invalid_degree_raises() -> None:
    df = pd.DataFrame({"a": [1.0, 2.0]})
    with pytest.raises(DataValidationError, match="degree"):
        apply_polynomial_features(df, columns=["a"], degree=1)


def test_polynomial_missing_column_raises() -> None:
    df = pd.DataFrame({"a": [1.0]})
    with pytest.raises(DataValidationError, match="not found"):
        apply_polynomial_features(df, columns=["z"], degree=2)


def test_polynomial_non_numeric_raises() -> None:
    df = pd.DataFrame({"a": ["x", "y"]})
    with pytest.raises(DataValidationError, match="Non-numeric"):
        apply_polynomial_features(df, columns=["a"], degree=2)


# ─────────────────────────────────────────────
# apply_binning
# ─────────────────────────────────────────────


def test_binning_integer_bins() -> None:
    df = pd.DataFrame({"score": [10.0, 40.0, 70.0, 95.0]})
    result = apply_binning(df, column="score", bins=4, output_column="score_bin")
    assert "score_bin" in result.columns
    assert result["score_bin"].isnull().sum() == 0


def test_binning_explicit_edges() -> None:
    df = pd.DataFrame({"score": [10.0, 50.0, 90.0]})
    result = apply_binning(df, column="score", bins=[0, 33, 66, 100], output_column="grade")
    assert "grade" in result.columns


def test_binning_with_labels() -> None:
    df = pd.DataFrame({"score": [10.0, 50.0, 90.0]})
    result = apply_binning(
        df,
        column="score",
        bins=[0, 33, 66, 100],
        output_column="grade",
        labels=["low", "mid", "high"],
    )
    assert result["grade"].tolist() == ["low", "mid", "high"]


def test_binning_missing_column_raises() -> None:
    df = pd.DataFrame({"a": [1.0]})
    with pytest.raises(DataValidationError, match="not found"):
        apply_binning(df, column="z", bins=3, output_column="z_bin")


def test_binning_non_numeric_raises() -> None:
    df = pd.DataFrame({"a": ["x", "y"]})
    with pytest.raises(DataValidationError, match="numeric"):
        apply_binning(df, column="a", bins=2, output_column="a_bin")


# ─────────────────────────────────────────────
# extract_datetime_features
# ─────────────────────────────────────────────


def test_datetime_extract_year_month() -> None:
    df = pd.DataFrame({"date": ["2024-03-15", "2023-07-04"]})
    result = extract_datetime_features(df, column="date", features=["year", "month"])
    assert "date_year" in result.columns
    assert "date_month" in result.columns
    assert result["date_year"].tolist() == [2024, 2023]
    assert result["date_month"].tolist() == [3, 7]


def test_datetime_extract_dayofweek() -> None:
    df = pd.DataFrame({"date": ["2024-01-01"]})
    result = extract_datetime_features(df, column="date", features=["dayofweek"])
    assert "date_dayofweek" in result.columns


def test_datetime_extract_all() -> None:
    df = pd.DataFrame({"date": ["2024-06-15 10:30:00"]})
    result = extract_datetime_features(df, column="date")
    for feat in ["year", "month", "day", "hour", "dayofweek"]:
        assert f"date_{feat}" in result.columns


def test_datetime_invalid_feature_raises() -> None:
    df = pd.DataFrame({"date": ["2024-01-01"]})
    with pytest.raises(DataValidationError, match="Invalid datetime features"):
        extract_datetime_features(df, column="date", features=["century"])


def test_datetime_missing_column_raises() -> None:
    df = pd.DataFrame({"a": [1]})
    with pytest.raises(DataValidationError, match="not found"):
        extract_datetime_features(df, column="date", features=["year"])
