"""
Tests: orchestration/base.py — PipelineResult
Version: 0.11.0
"""

import pandas as pd
import pytest
from pipelineforge.orchestration.base import PipelineResult
from sklearn.ensemble import RandomForestClassifier


@pytest.fixture
def sample_result() -> PipelineResult:
    model = RandomForestClassifier(n_estimators=5, random_state=42)
    X = pd.DataFrame({"f1": [1.0, 2.0], "f2": [3.0, 4.0]})
    y = pd.Series([0, 1])
    model.fit(X, y)

    return PipelineResult(
        model=model,
        metrics={"accuracy": 1.0, "f1": 1.0},
        X_train=X,
        X_test=X,
        y_train=y,
        y_test=y,
        predictions=pd.Series([0, 1]),
        model_type="random_forest",
        target_column="target",
    )


def test_pipeline_result_instantiation(sample_result) -> None:
    assert sample_result is not None


def test_pipeline_result_model_field(sample_result) -> None:
    assert isinstance(sample_result.model, RandomForestClassifier)


def test_pipeline_result_metrics_field(sample_result) -> None:
    assert isinstance(sample_result.metrics, dict)
    assert "accuracy" in sample_result.metrics


def test_pipeline_result_predictions_is_series(sample_result) -> None:
    assert isinstance(sample_result.predictions, pd.Series)


def test_pipeline_result_model_type_is_str(sample_result) -> None:
    assert isinstance(sample_result.model_type, str)
    assert sample_result.model_type == "random_forest"


def test_pipeline_result_target_column_is_str(sample_result) -> None:
    assert isinstance(sample_result.target_column, str)
    assert sample_result.target_column == "target"


def test_pipeline_result_slots_no_extra_attrs(sample_result) -> None:
    with pytest.raises(AttributeError):
        sample_result.nonexistent_field = "bad"


def test_pipeline_result_X_train_is_dataframe(sample_result) -> None:
    assert isinstance(sample_result.X_train, pd.DataFrame)


def test_pipeline_result_y_test_is_series(sample_result) -> None:
    assert isinstance(sample_result.y_test, pd.Series)
