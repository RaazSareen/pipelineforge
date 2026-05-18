"""
Tests: evaluation/classification.py
Version: 0.9.0
"""

import pandas as pd
import pytest
from pipelineforge.evaluation.classification import (
    compute_accuracy,
    compute_all_classification_metrics,
    compute_classification_report,
    compute_confusion_matrix,
    compute_f1,
    compute_precision,
    compute_recall,
    compute_roc_auc,
)
from pipelineforge.exceptions import DataValidationError


@pytest.fixture
def binary_preds():
    y_true = pd.Series([0, 1, 0, 1, 0, 1, 0, 1])
    y_pred = pd.Series([0, 1, 0, 1, 0, 0, 1, 1])
    return y_true, y_pred


@pytest.fixture
def perfect_preds():
    y_true = pd.Series([0, 1, 0, 1])
    y_pred = pd.Series([0, 1, 0, 1])
    return y_true, y_pred


# ─────────────────────────────────────────────
# compute_accuracy
# ─────────────────────────────────────────────


def test_accuracy_perfect(perfect_preds) -> None:
    y_true, y_pred = perfect_preds
    assert compute_accuracy(y_true, y_pred) == pytest.approx(1.0)


def test_accuracy_partial(binary_preds) -> None:
    y_true, y_pred = binary_preds
    score = compute_accuracy(y_true, y_pred)
    assert 0.0 <= score <= 1.0


def test_accuracy_empty_raises() -> None:
    with pytest.raises(DataValidationError, match="empty"):
        compute_accuracy(pd.Series([], dtype=int), pd.Series([], dtype=int))


def test_accuracy_mismatched_length_raises() -> None:
    with pytest.raises(DataValidationError, match="same length"):
        compute_accuracy(pd.Series([0, 1]), pd.Series([0]))


# ─────────────────────────────────────────────
# compute_precision
# ─────────────────────────────────────────────


def test_precision_perfect(perfect_preds) -> None:
    y_true, y_pred = perfect_preds
    assert compute_precision(y_true, y_pred) == pytest.approx(1.0)


def test_precision_invalid_average_raises(binary_preds) -> None:
    y_true, y_pred = binary_preds
    with pytest.raises(DataValidationError, match="Invalid average"):
        compute_precision(y_true, y_pred, average="bad")


# ─────────────────────────────────────────────
# compute_recall
# ─────────────────────────────────────────────


def test_recall_perfect(perfect_preds) -> None:
    y_true, y_pred = perfect_preds
    assert compute_recall(y_true, y_pred) == pytest.approx(1.0)


def test_recall_macro(binary_preds) -> None:
    y_true, y_pred = binary_preds
    score = compute_recall(y_true, y_pred, average="macro")
    assert 0.0 <= score <= 1.0


# ─────────────────────────────────────────────
# compute_f1
# ─────────────────────────────────────────────


def test_f1_perfect(perfect_preds) -> None:
    y_true, y_pred = perfect_preds
    assert compute_f1(y_true, y_pred) == pytest.approx(1.0)


def test_f1_partial(binary_preds) -> None:
    y_true, y_pred = binary_preds
    score = compute_f1(y_true, y_pred)
    assert 0.0 <= score <= 1.0


# ─────────────────────────────────────────────
# compute_roc_auc
# ─────────────────────────────────────────────


def test_roc_auc_binary() -> None:
    y_true = pd.Series([0, 0, 1, 1])
    y_score = pd.Series([0.1, 0.4, 0.6, 0.9])
    score = compute_roc_auc(y_true, y_score)
    assert 0.0 <= score <= 1.0


def test_roc_auc_perfect_binary() -> None:
    y_true = pd.Series([0, 0, 1, 1])
    y_score = pd.Series([0.0, 0.1, 0.9, 1.0])
    score = compute_roc_auc(y_true, y_score)
    assert score == pytest.approx(1.0)


def test_roc_auc_empty_raises() -> None:
    with pytest.raises(DataValidationError, match="empty"):
        compute_roc_auc(
            pd.Series([], dtype=int),
            pd.Series([], dtype=float),
        )


def test_roc_auc_single_class_raises() -> None:
    y_true = pd.Series([1, 1, 1, 1])
    y_score = pd.Series([0.9, 0.8, 0.7, 0.6])
    with pytest.raises(DataValidationError, match="ROC AUC"):
        compute_roc_auc(y_true, y_score)


# ─────────────────────────────────────────────
# compute_confusion_matrix
# ─────────────────────────────────────────────


def test_confusion_matrix_shape(binary_preds) -> None:
    y_true, y_pred = binary_preds
    result = compute_confusion_matrix(y_true, y_pred)
    assert isinstance(result, pd.DataFrame)
    assert result.shape == (2, 2)


def test_confusion_matrix_perfect(perfect_preds) -> None:
    y_true, y_pred = perfect_preds
    result = compute_confusion_matrix(y_true, y_pred)
    assert result.values.trace() == len(y_true)


def test_confusion_matrix_empty_raises() -> None:
    with pytest.raises(DataValidationError, match="empty"):
        compute_confusion_matrix(
            pd.Series([], dtype=int),
            pd.Series([], dtype=int),
        )


# ─────────────────────────────────────────────
# compute_classification_report
# ─────────────────────────────────────────────


def test_classification_report_returns_string(binary_preds) -> None:
    y_true, y_pred = binary_preds
    report = compute_classification_report(y_true, y_pred)
    assert isinstance(report, str)
    assert "precision" in report


def test_classification_report_empty_raises() -> None:
    with pytest.raises(DataValidationError, match="empty"):
        compute_classification_report(
            pd.Series([], dtype=int),
            pd.Series([], dtype=int),
        )


# ─────────────────────────────────────────────
# compute_all_classification_metrics
# ─────────────────────────────────────────────


def test_all_clf_metrics_keys(binary_preds) -> None:
    y_true, y_pred = binary_preds
    metrics = compute_all_classification_metrics(y_true, y_pred)
    assert set(metrics.keys()) == {"accuracy", "precision", "recall", "f1"}


def test_all_clf_metrics_values_in_range(binary_preds) -> None:
    y_true, y_pred = binary_preds
    metrics = compute_all_classification_metrics(y_true, y_pred)
    for val in metrics.values():
        assert 0.0 <= val <= 1.0


def test_all_clf_metrics_perfect(perfect_preds) -> None:
    y_true, y_pred = perfect_preds
    metrics = compute_all_classification_metrics(y_true, y_pred)
    assert metrics["accuracy"] == pytest.approx(1.0)
    assert metrics["f1"] == pytest.approx(1.0)
