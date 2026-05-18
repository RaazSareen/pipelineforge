"""
Tests: evaluation/regression.py
Version: 0.9.0
"""

import math

import pandas as pd
import pytest
from pipelineforge.evaluation.regression import (
    compute_all_regression_metrics,
    compute_mae,
    compute_mape,
    compute_mse,
    compute_r2,
    compute_rmse,
)
from pipelineforge.exceptions import DataValidationError


@pytest.fixture
def perfect_reg():
    y_true = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    y_pred = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    return y_true, y_pred


@pytest.fixture
def noisy_reg():
    y_true = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    y_pred = pd.Series([1.1, 2.2, 2.9, 3.8, 5.1])
    return y_true, y_pred


# ─────────────────────────────────────────────
# compute_mae
# ─────────────────────────────────────────────


def test_mae_perfect(perfect_reg) -> None:
    y_true, y_pred = perfect_reg
    assert compute_mae(y_true, y_pred) == pytest.approx(0.0)


def test_mae_positive(noisy_reg) -> None:
    y_true, y_pred = noisy_reg
    assert compute_mae(y_true, y_pred) > 0.0


def test_mae_empty_raises() -> None:
    with pytest.raises(DataValidationError, match="empty"):
        compute_mae(pd.Series([], dtype=float), pd.Series([], dtype=float))


def test_mae_mismatched_raises() -> None:
    with pytest.raises(DataValidationError, match="same length"):
        compute_mae(pd.Series([1.0, 2.0]), pd.Series([1.0]))


# ─────────────────────────────────────────────
# compute_mse
# ─────────────────────────────────────────────


def test_mse_perfect(perfect_reg) -> None:
    y_true, y_pred = perfect_reg
    assert compute_mse(y_true, y_pred) == pytest.approx(0.0)


def test_mse_positive(noisy_reg) -> None:
    y_true, y_pred = noisy_reg
    assert compute_mse(y_true, y_pred) > 0.0


# ─────────────────────────────────────────────
# compute_rmse
# ─────────────────────────────────────────────


def test_rmse_perfect(perfect_reg) -> None:
    y_true, y_pred = perfect_reg
    assert compute_rmse(y_true, y_pred) == pytest.approx(0.0)


def test_rmse_equals_sqrt_mse(noisy_reg) -> None:
    y_true, y_pred = noisy_reg
    mse = compute_mse(y_true, y_pred)
    rmse = compute_rmse(y_true, y_pred)
    assert rmse == pytest.approx(math.sqrt(mse))


# ─────────────────────────────────────────────
# compute_r2
# ─────────────────────────────────────────────


def test_r2_perfect(perfect_reg) -> None:
    y_true, y_pred = perfect_reg
    assert compute_r2(y_true, y_pred) == pytest.approx(1.0)


def test_r2_is_float(noisy_reg) -> None:
    y_true, y_pred = noisy_reg
    assert isinstance(compute_r2(y_true, y_pred), float)


# ─────────────────────────────────────────────
# compute_mape
# ─────────────────────────────────────────────


def test_mape_perfect(perfect_reg) -> None:
    y_true, y_pred = perfect_reg
    assert compute_mape(y_true, y_pred) == pytest.approx(0.0)


def test_mape_positive(noisy_reg) -> None:
    y_true, y_pred = noisy_reg
    assert compute_mape(y_true, y_pred) > 0.0


def test_mape_zero_in_y_true_raises() -> None:
    y_true = pd.Series([1.0, 0.0, 3.0])
    y_pred = pd.Series([1.1, 0.5, 2.9])
    with pytest.raises(DataValidationError, match="zero"):
        compute_mape(y_true, y_pred)


# ─────────────────────────────────────────────
# compute_all_regression_metrics
# ─────────────────────────────────────────────


def test_all_reg_metrics_keys(noisy_reg) -> None:
    y_true, y_pred = noisy_reg
    metrics = compute_all_regression_metrics(y_true, y_pred)
    assert "mae" in metrics
    assert "mse" in metrics
    assert "rmse" in metrics
    assert "r2" in metrics
    assert "mape" in metrics


def test_all_reg_metrics_non_negative(noisy_reg) -> None:
    y_true, y_pred = noisy_reg
    metrics = compute_all_regression_metrics(y_true, y_pred)
    assert metrics["mae"] >= 0.0
    assert metrics["mse"] >= 0.0
    assert metrics["rmse"] >= 0.0


def test_all_reg_metrics_mape_skipped_on_zeros() -> None:
    y_true = pd.Series([1.0, 0.0, 3.0])
    y_pred = pd.Series([1.1, 0.5, 2.9])
    metrics = compute_all_regression_metrics(y_true, y_pred)
    assert "mape" not in metrics


def test_all_reg_metrics_perfect(perfect_reg) -> None:
    y_true, y_pred = perfect_reg
    metrics = compute_all_regression_metrics(y_true, y_pred)
    assert metrics["mae"] == pytest.approx(0.0)
    assert metrics["r2"] == pytest.approx(1.0)
