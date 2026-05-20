"""
Tests: orchestration/regression_pipeline.py
Version: 0.11.0
"""

import pandas as pd
import pytest
from pipelineforge.exceptions import ConfigurationError, DataValidationError
from pipelineforge.orchestration.base import PipelineResult
from pipelineforge.orchestration.regression_pipeline import run_regression_pipeline


@pytest.fixture
def reg_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "f1": [float(i) for i in range(1, 21)],
            "f2": [float(i) * 0.5 for i in range(1, 21)],
            "target": [float(i) * 1.1 for i in range(1, 21)],
        }
    )


# ─────────────────────────────────────────────
# basic execution
# ─────────────────────────────────────────────


def test_reg_pipeline_returns_result(reg_df) -> None:
    result = run_regression_pipeline(reg_df, target_column="target")
    assert isinstance(result, PipelineResult)


def test_reg_pipeline_metrics_keys(reg_df) -> None:
    result = run_regression_pipeline(reg_df, target_column="target")
    assert "mae" in result.metrics
    assert "mse" in result.metrics
    assert "rmse" in result.metrics
    assert "r2" in result.metrics


def test_reg_pipeline_metrics_non_negative(reg_df) -> None:
    result = run_regression_pipeline(reg_df, target_column="target")
    assert result.metrics["mae"] >= 0.0
    assert result.metrics["mse"] >= 0.0
    assert result.metrics["rmse"] >= 0.0


def test_reg_pipeline_predictions_length(reg_df) -> None:
    result = run_regression_pipeline(reg_df, target_column="target", test_size=0.2)
    assert len(result.predictions) == len(result.y_test)


def test_reg_pipeline_predictions_is_series(reg_df) -> None:
    result = run_regression_pipeline(reg_df, target_column="target")
    assert isinstance(result.predictions, pd.Series)


def test_reg_pipeline_model_type_stored(reg_df) -> None:
    result = run_regression_pipeline(reg_df, target_column="target", model_type="linear_regression")
    assert result.model_type == "linear_regression"


def test_reg_pipeline_target_column_stored(reg_df) -> None:
    result = run_regression_pipeline(reg_df, target_column="target")
    assert result.target_column == "target"


def test_reg_pipeline_random_forest(reg_df) -> None:
    result = run_regression_pipeline(
        reg_df,
        target_column="target",
        model_type="random_forest",
        model_kwargs={"n_estimators": 10, "random_state": 42},
    )
    assert isinstance(result, PipelineResult)


def test_reg_pipeline_ridge(reg_df) -> None:
    result = run_regression_pipeline(
        reg_df,
        target_column="target",
        model_type="ridge",
        model_kwargs={"alpha": 1.0},
    )
    assert isinstance(result, PipelineResult)


# ─────────────────────────────────────────────
# edge cases
# ─────────────────────────────────────────────


def test_reg_pipeline_empty_dataframe_raises() -> None:
    with pytest.raises(DataValidationError, match="empty"):
        run_regression_pipeline(pd.DataFrame(), target_column="target")


def test_reg_pipeline_missing_target_raises(reg_df) -> None:
    with pytest.raises(DataValidationError, match="Target column not found"):
        run_regression_pipeline(reg_df, target_column="missing")


def test_reg_pipeline_invalid_model_type_raises(reg_df) -> None:
    with pytest.raises(DataValidationError, match="Unknown regressor"):
        run_regression_pipeline(reg_df, target_column="target", model_type="svm")


def test_reg_pipeline_invalid_test_size_raises(reg_df) -> None:
    with pytest.raises(ConfigurationError, match="test_size"):
        run_regression_pipeline(reg_df, target_column="target", test_size=0.0)
