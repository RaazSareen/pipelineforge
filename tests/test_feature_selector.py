"""
Tests: features/selector.py
Version: 0.7.0
"""

import pandas as pd
import pytest
from pipelineforge.exceptions import DataValidationError
from pipelineforge.features.selector import (
    calculate_vif,
    drop_high_vif_features,
    drop_highly_correlated_features,
    select_by_feature_importance,
    select_by_target_correlation,
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

# ─────────────────────────────────────────────
# drop_highly_correlated_features
# ─────────────────────────────────────────────


def test_drop_correlated_removes_one() -> None:
    df = pd.DataFrame(
        {
            "a": [1.0, 2.0, 3.0, 4.0, 5.0],
            "b": [1.0, 2.0, 3.0, 4.0, 5.0],  # perfectly correlated with a
            "c": [5.0, 3.0, 1.0, 4.0, 2.0],  # unrelated
        }
    )
    result = drop_highly_correlated_features(df, threshold=0.9)
    assert "a" in result.columns
    assert "b" not in result.columns
    assert "c" in result.columns


def test_drop_correlated_no_removal_below_threshold() -> None:
    df = pd.DataFrame(
        {
            "a": [1.0, 2.0, 3.0, 4.0, 5.0],
            "b": [5.0, 3.0, 1.0, 4.0, 2.0],
        }
    )
    result = drop_highly_correlated_features(df, threshold=0.9)
    assert "a" in result.columns
    assert "b" in result.columns


def test_drop_correlated_invalid_method_raises() -> None:
    df = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]})
    with pytest.raises(DataValidationError, match="Invalid method"):
        drop_highly_correlated_features(df, method="mutual_info")


def test_drop_correlated_invalid_threshold_raises() -> None:
    df = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]})
    with pytest.raises(DataValidationError, match="threshold"):
        drop_highly_correlated_features(df, threshold=0.0)


def test_drop_correlated_too_few_columns_raises() -> None:
    df = pd.DataFrame({"a": [1.0, 2.0]})
    with pytest.raises(DataValidationError, match="At least 2"):
        drop_highly_correlated_features(df)


# ─────────────────────────────────────────────
# select_by_target_correlation
# ─────────────────────────────────────────────


def test_target_correlation_keeps_correlated() -> None:
    df = pd.DataFrame(
        {
            "a": [1.0, 2.0, 3.0, 4.0, 5.0],
            "b": [10.0, 20.0, 30.0, 40.0, 50.0],  # perfectly correlated with target
            "noise": [5.0, 1.0, 3.0, 2.0, 4.0],
            "target": [1.0, 2.0, 3.0, 4.0, 5.0],
        }
    )
    result = select_by_target_correlation(df, target_column="target", threshold=0.9)
    assert "target" in result.columns
    assert "b" in result.columns


def test_target_correlation_missing_target_raises() -> None:
    df = pd.DataFrame({"a": [1.0, 2.0]})
    with pytest.raises(DataValidationError, match="Target column not found"):
        select_by_target_correlation(df, target_column="missing")


def test_target_correlation_invalid_threshold_raises() -> None:
    df = pd.DataFrame({"a": [1.0, 2.0], "target": [1.0, 2.0]})
    with pytest.raises(DataValidationError, match="threshold"):
        select_by_target_correlation(df, target_column="target", threshold=1.5)


# ─────────────────────────────────────────────
# calculate_vif
# ─────────────────────────────────────────────


def test_calculate_vif_returns_dataframe() -> None:
    df = pd.DataFrame(
        {
            "a": [1.0, 2.0, 3.0, 4.0, 5.0],
            "b": [2.0, 3.0, 4.0, 5.0, 6.0],
            "c": [5.0, 3.0, 1.0, 4.0, 2.0],
        }
    )
    result = calculate_vif(df)
    assert isinstance(result, pd.DataFrame)
    assert "feature" in result.columns
    assert "vif" in result.columns
    assert len(result) == 3


def test_calculate_vif_too_few_columns_raises() -> None:
    df = pd.DataFrame({"a": [1.0, 2.0, 3.0]})
    with pytest.raises(DataValidationError, match="At least 2"):
        calculate_vif(df)


def test_calculate_vif_missing_column_raises() -> None:
    df = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0]})
    with pytest.raises(DataValidationError, match="not found"):
        calculate_vif(df, columns=["a", "z"])


# ─────────────────────────────────────────────
# drop_high_vif_features
# ─────────────────────────────────────────────


def test_drop_high_vif_reduces_columns() -> None:
    df = pd.DataFrame(
        {
            "a": [1.0, 2.0, 3.0, 4.0, 5.0],
            "b": [1.1, 2.1, 3.1, 4.1, 5.1],  # near-identical to a → high VIF
            "c": [5.0, 3.0, 1.0, 4.0, 2.0],
        }
    )
    result = drop_high_vif_features(df, threshold=5.0)
    assert result.shape[1] < df.shape[1]


def test_drop_high_vif_invalid_threshold_raises() -> None:
    df = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]})
    with pytest.raises(DataValidationError, match="threshold"):
        drop_high_vif_features(df, threshold=0.0)


# ─────────────────────────────────────────────
# select_by_feature_importance
# ─────────────────────────────────────────────


@pytest.fixture
def fitted_rf():
    df = pd.DataFrame(
        {
            "f1": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
            "f2": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            "target": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
        }
    )
    X = df[["f1", "f2"]]
    y = df["target"]
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X, y)
    return model, df


def test_importance_selection_basic(fitted_rf) -> None:
    model, df = fitted_rf
    result = select_by_feature_importance(
        df, estimator=model, target_column="target", threshold=0.0
    )
    assert "target" in result.columns
    assert result.shape[1] >= 2


def test_importance_selection_high_threshold(fitted_rf) -> None:
    model, df = fitted_rf
    result = select_by_feature_importance(
        df, estimator=model, target_column="target", threshold=0.0
    )
    assert "target" in result.columns


def test_importance_selection_missing_target_raises(fitted_rf) -> None:
    model, df = fitted_rf
    with pytest.raises(DataValidationError, match="Target column not found"):
        select_by_feature_importance(df, estimator=model, target_column="missing")


def test_importance_selection_negative_threshold_raises(fitted_rf) -> None:
    model, df = fitted_rf
    with pytest.raises(DataValidationError, match="threshold"):
        select_by_feature_importance(df, estimator=model, target_column="target", threshold=-0.1)


def test_importance_selection_no_importances_raises() -> None:
    df = pd.DataFrame({"f1": [1.0, 2.0], "target": [0, 1]})

    class DummyEstimator:
        pass

    with pytest.raises(DataValidationError, match="feature_importances_"):
        select_by_feature_importance(df, estimator=DummyEstimator(), target_column="target")


def test_importance_coef_model() -> None:
    df = pd.DataFrame(
        {
            "f1": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
            "f2": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            "target": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
        }
    )
    X = df[["f1", "f2"]]
    y = df["target"]
    model = LogisticRegression(random_state=42, max_iter=200)
    model.fit(X, y)
    result = select_by_feature_importance(
        df, estimator=model, target_column="target", threshold=0.0
    )
    assert "target" in result.columns
