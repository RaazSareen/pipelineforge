"""
Tests: models/regressor.py
Version: 0.8.0
"""

import pandas as pd
import pytest
from pipelineforge.exceptions import DataValidationError
from pipelineforge.model_selection.splitter import train_test_split_data
from pipelineforge.models.regressor import (
    build_regressor,
    get_available_regressors,
    train_regressor,
)
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge


@pytest.fixture
def reg_split():
    df = pd.DataFrame(
        {
            "f1": [float(i) for i in range(1, 21)],
            "f2": [float(i) * 0.5 for i in range(1, 21)],
            "target": [float(i) * 1.1 for i in range(1, 21)],
        }
    )
    X_train, X_test, y_train, y_test = train_test_split_data(
        df, target_column="target", test_size=0.2, random_state=42
    )
    return X_train, X_test, y_train, y_test


# ─────────────────────────────────────────────
# get_available_regressors
# ─────────────────────────────────────────────


def test_get_available_regressors_returns_list() -> None:
    result = get_available_regressors()
    assert isinstance(result, list)
    assert len(result) > 0


def test_get_available_regressors_contains_expected() -> None:
    result = get_available_regressors()
    assert "random_forest" in result
    assert "linear_regression" in result
    assert "gradient_boosting" in result
    assert "ridge" in result


# ─────────────────────────────────────────────
# build_regressor
# ─────────────────────────────────────────────


def test_build_random_forest() -> None:
    model = build_regressor("random_forest", n_estimators=10)
    assert isinstance(model, RandomForestRegressor)


def test_build_linear_regression() -> None:
    model = build_regressor("linear_regression")
    assert isinstance(model, LinearRegression)


def test_build_gradient_boosting() -> None:
    model = build_regressor("gradient_boosting", n_estimators=10)
    assert isinstance(model, GradientBoostingRegressor)


def test_build_ridge() -> None:
    model = build_regressor("ridge", alpha=1.0)
    assert isinstance(model, Ridge)


def test_build_regressor_kwargs_passed() -> None:
    model = build_regressor("random_forest", n_estimators=7, random_state=0)
    assert model.n_estimators == 7
    assert model.random_state == 0


def test_build_regressor_invalid_raises() -> None:
    with pytest.raises(DataValidationError, match="Unknown regressor"):
        build_regressor("svm")


# ─────────────────────────────────────────────
# train_regressor
# ─────────────────────────────────────────────


def test_train_regressor_returns_fitted(reg_split) -> None:
    X_train, X_test, y_train, y_test = reg_split
    model = build_regressor("linear_regression")
    fitted = train_regressor(model, X_train, y_train)
    preds = fitted.predict(X_test)
    assert len(preds) == len(y_test)


def test_train_regressor_random_forest(reg_split) -> None:
    X_train, X_test, y_train, y_test = reg_split
    model = build_regressor("random_forest", n_estimators=10, random_state=42)
    fitted = train_regressor(model, X_train, y_train)
    preds = fitted.predict(X_test)
    assert len(preds) == len(y_test)


def test_train_regressor_ridge(reg_split) -> None:
    X_train, X_test, y_train, y_test = reg_split
    model = build_regressor("ridge", alpha=0.5)
    fitted = train_regressor(model, X_train, y_train)
    preds = fitted.predict(X_test)
    assert len(preds) == len(y_test)


def test_train_regressor_empty_X_raises() -> None:
    model = build_regressor("linear_regression")
    with pytest.raises(DataValidationError, match="X_train is empty"):
        train_regressor(model, pd.DataFrame(), pd.Series([], dtype=float))


def test_train_regressor_empty_y_raises() -> None:
    model = build_regressor("linear_regression")
    X = pd.DataFrame({"f1": [1.0, 2.0]})
    with pytest.raises(DataValidationError, match="y_train is empty"):
        train_regressor(model, X, pd.Series([], dtype=float))


def test_train_regressor_mismatched_lengths_raises() -> None:
    model = build_regressor("linear_regression")
    X = pd.DataFrame({"f1": [1.0, 2.0, 3.0]})
    y = pd.Series([1.0, 2.0])
    with pytest.raises(DataValidationError, match="same length"):
        train_regressor(model, X, y)
