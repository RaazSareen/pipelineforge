"""
Tests: models/classifier.py
Version: 0.8.0
"""

import pandas as pd
import pytest
from pipelineforge.exceptions import DataValidationError
from pipelineforge.model_selection.splitter import train_test_split_data
from pipelineforge.models.classifier import (
    build_classifier,
    get_available_classifiers,
    train_classifier,
)
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression


@pytest.fixture
def clf_split():
    df = pd.DataFrame(
        {
            "f1": [float(i) for i in range(1, 21)],
            "f2": [float(i) * 0.1 for i in range(1, 21)],
            "target": [i % 2 for i in range(1, 21)],
        }
    )
    X_train, X_test, y_train, y_test = train_test_split_data(
        df, target_column="target", test_size=0.2, random_state=42
    )
    return X_train, X_test, y_train, y_test


# ─────────────────────────────────────────────
# get_available_classifiers
# ─────────────────────────────────────────────


def test_get_available_classifiers_returns_list() -> None:
    result = get_available_classifiers()
    assert isinstance(result, list)
    assert len(result) > 0


def test_get_available_classifiers_contains_expected() -> None:
    result = get_available_classifiers()
    assert "random_forest" in result
    assert "logistic_regression" in result
    assert "gradient_boosting" in result


# ─────────────────────────────────────────────
# build_classifier
# ─────────────────────────────────────────────


def test_build_random_forest() -> None:
    model = build_classifier("random_forest", n_estimators=10)
    assert isinstance(model, RandomForestClassifier)


def test_build_logistic_regression() -> None:
    model = build_classifier("logistic_regression", max_iter=200)
    assert isinstance(model, LogisticRegression)


def test_build_gradient_boosting() -> None:
    model = build_classifier("gradient_boosting", n_estimators=10)
    assert isinstance(model, GradientBoostingClassifier)


def test_build_classifier_kwargs_passed() -> None:
    model = build_classifier("random_forest", n_estimators=7, random_state=0)
    assert model.n_estimators == 7
    assert model.random_state == 0


def test_build_classifier_invalid_raises() -> None:
    with pytest.raises(DataValidationError, match="Unknown classifier"):
        build_classifier("svm")


# ─────────────────────────────────────────────
# train_classifier
# ─────────────────────────────────────────────


def test_train_classifier_returns_fitted(clf_split) -> None:
    X_train, X_test, y_train, y_test = clf_split
    model = build_classifier("random_forest", n_estimators=10, random_state=42)
    fitted = train_classifier(model, X_train, y_train)
    preds = fitted.predict(X_test)
    assert len(preds) == len(y_test)


def test_train_classifier_logistic(clf_split) -> None:
    X_train, X_test, y_train, y_test = clf_split
    model = build_classifier("logistic_regression", max_iter=200)
    fitted = train_classifier(model, X_train, y_train)
    preds = fitted.predict(X_test)
    assert len(preds) == len(y_test)


def test_train_classifier_gradient_boosting(clf_split) -> None:
    X_train, X_test, y_train, y_test = clf_split
    model = build_classifier("gradient_boosting", n_estimators=10)
    fitted = train_classifier(model, X_train, y_train)
    preds = fitted.predict(X_test)
    assert len(preds) == len(y_test)


def test_train_classifier_empty_X_raises() -> None:
    model = build_classifier("random_forest", n_estimators=10)
    with pytest.raises(DataValidationError, match="X_train is empty"):
        train_classifier(model, pd.DataFrame(), pd.Series([], dtype=float))


def test_train_classifier_empty_y_raises() -> None:
    model = build_classifier("random_forest", n_estimators=10)
    X = pd.DataFrame({"f1": [1.0, 2.0]})
    with pytest.raises(DataValidationError, match="y_train is empty"):
        train_classifier(model, X, pd.Series([], dtype=float))


def test_train_classifier_mismatched_lengths_raises() -> None:
    model = build_classifier("random_forest", n_estimators=10)
    X = pd.DataFrame({"f1": [1.0, 2.0, 3.0]})
    y = pd.Series([0, 1])
    with pytest.raises(DataValidationError, match="same length"):
        train_classifier(model, X, y)
