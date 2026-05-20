"""
Tests: orchestration/classification_pipeline.py
Version: 0.11.0
"""

import pandas as pd
import pytest
from pipelineforge.exceptions import ConfigurationError, DataValidationError
from pipelineforge.orchestration.base import PipelineResult
from pipelineforge.orchestration.classification_pipeline import (
    run_classification_pipeline,
)


@pytest.fixture
def clf_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "f1": [float(i) for i in range(1, 21)],
            "f2": [float(i) * 0.1 for i in range(1, 21)],
            "target": [i % 2 for i in range(1, 21)],
        }
    )


# ─────────────────────────────────────────────
# basic execution
# ─────────────────────────────────────────────


def test_clf_pipeline_returns_result(clf_df) -> None:
    result = run_classification_pipeline(clf_df, target_column="target")
    assert isinstance(result, PipelineResult)


def test_clf_pipeline_metrics_keys(clf_df) -> None:
    result = run_classification_pipeline(clf_df, target_column="target")
    assert set(result.metrics.keys()) == {"accuracy", "precision", "recall", "f1"}


def test_clf_pipeline_metrics_in_range(clf_df) -> None:
    result = run_classification_pipeline(clf_df, target_column="target")
    for val in result.metrics.values():
        assert 0.0 <= val <= 1.0


def test_clf_pipeline_predictions_length(clf_df) -> None:
    result = run_classification_pipeline(clf_df, target_column="target", test_size=0.2)
    assert len(result.predictions) == len(result.y_test)


def test_clf_pipeline_predictions_is_series(clf_df) -> None:
    result = run_classification_pipeline(clf_df, target_column="target")
    assert isinstance(result.predictions, pd.Series)


def test_clf_pipeline_model_type_stored(clf_df) -> None:
    result = run_classification_pipeline(clf_df, target_column="target", model_type="random_forest")
    assert result.model_type == "random_forest"


def test_clf_pipeline_target_column_stored(clf_df) -> None:
    result = run_classification_pipeline(clf_df, target_column="target")
    assert result.target_column == "target"


def test_clf_pipeline_logistic_regression(clf_df) -> None:
    result = run_classification_pipeline(
        clf_df,
        target_column="target",
        model_type="logistic_regression",
        model_kwargs={"max_iter": 200},
    )
    assert isinstance(result, PipelineResult)


# ─────────────────────────────────────────────
# edge cases
# ─────────────────────────────────────────────


def test_clf_pipeline_empty_dataframe_raises() -> None:
    with pytest.raises(DataValidationError, match="empty"):
        run_classification_pipeline(pd.DataFrame(), target_column="target")


def test_clf_pipeline_missing_target_raises(clf_df) -> None:
    with pytest.raises(DataValidationError, match="Target column not found"):
        run_classification_pipeline(clf_df, target_column="missing")


def test_clf_pipeline_invalid_model_type_raises(clf_df) -> None:
    with pytest.raises(DataValidationError, match="Unknown classifier"):
        run_classification_pipeline(clf_df, target_column="target", model_type="svm")


def test_clf_pipeline_invalid_test_size_raises(clf_df) -> None:
    with pytest.raises(ConfigurationError, match="test_size"):
        run_classification_pipeline(clf_df, target_column="target", test_size=1.5)
