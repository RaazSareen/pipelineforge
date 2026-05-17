"""
Tests: model_selection/splitter.py
Version: 0.7.0
"""

import pandas as pd
import pytest
from pipelineforge.exceptions import ConfigurationError, DataValidationError
from pipelineforge.model_selection.splitter import (
    kfold_split,
    stratified_split,
    timeseries_split,
    train_test_split_data,
)


@pytest.fixture
def sample_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "f1": [float(i) for i in range(1, 21)],
            "f2": [float(i) * 0.1 for i in range(1, 21)],
            "target": [i % 2 for i in range(1, 21)],
        }
    )


@pytest.fixture
def sample_reg_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "f1": [float(i) for i in range(1, 21)],
            "f2": [float(i) * 0.5 for i in range(1, 21)],
            "target": [float(i) * 1.1 for i in range(1, 21)],
        }
    )


# ─────────────────────────────────────────────
# train_test_split_data
# ─────────────────────────────────────────────


def test_tts_basic_split_sizes(sample_df) -> None:
    X_train, X_test, y_train, y_test = train_test_split_data(
        sample_df, target_column="target", test_size=0.2
    )
    assert len(X_train) == 16
    assert len(X_test) == 4
    assert len(y_train) == 16
    assert len(y_test) == 4


def test_tts_target_excluded_from_X(sample_df) -> None:
    X_train, X_test, _, _ = train_test_split_data(sample_df, target_column="target")
    assert "target" not in X_train.columns
    assert "target" not in X_test.columns


def test_tts_reproducible(sample_df) -> None:
    X_train_1, _, _, _ = train_test_split_data(sample_df, target_column="target", random_state=42)
    X_train_2, _, _, _ = train_test_split_data(sample_df, target_column="target", random_state=42)
    assert X_train_1.index.tolist() == X_train_2.index.tolist()


def test_tts_different_random_state(sample_df) -> None:
    X_train_1, _, _, _ = train_test_split_data(sample_df, target_column="target", random_state=0)
    X_train_2, _, _, _ = train_test_split_data(sample_df, target_column="target", random_state=99)
    assert X_train_1.index.tolist() != X_train_2.index.tolist()


def test_tts_empty_dataframe_raises() -> None:
    with pytest.raises(DataValidationError, match="empty"):
        train_test_split_data(pd.DataFrame(), target_column="target")


def test_tts_missing_target_raises(sample_df) -> None:
    with pytest.raises(DataValidationError, match="Target column not found"):
        train_test_split_data(sample_df, target_column="missing")


def test_tts_invalid_test_size_raises(sample_df) -> None:
    with pytest.raises(ConfigurationError, match="test_size"):
        train_test_split_data(sample_df, target_column="target", test_size=1.5)


def test_tts_test_size_zero_raises(sample_df) -> None:
    with pytest.raises(ConfigurationError, match="test_size"):
        train_test_split_data(sample_df, target_column="target", test_size=0.0)


# ─────────────────────────────────────────────
# stratified_split
# ─────────────────────────────────────────────


def test_stratified_split_sizes(sample_df) -> None:
    X_train, X_test, y_train, y_test = stratified_split(
        sample_df, target_column="target", test_size=0.2
    )
    assert len(X_train) == 16
    assert len(X_test) == 4


def test_stratified_split_class_balance(sample_df) -> None:
    _, _, _, y_test = stratified_split(
        sample_df, target_column="target", test_size=0.2, random_state=42
    )
    counts = y_test.value_counts()
    assert counts[0] == counts[1]


def test_stratified_split_missing_target_raises(sample_df) -> None:
    with pytest.raises(DataValidationError, match="Target column not found"):
        stratified_split(sample_df, target_column="missing")


def test_stratified_split_invalid_test_size_raises(sample_df) -> None:
    with pytest.raises(ConfigurationError, match="test_size"):
        stratified_split(sample_df, target_column="target", test_size=0.0)


def test_stratified_split_empty_raises() -> None:
    with pytest.raises(DataValidationError, match="empty"):
        stratified_split(pd.DataFrame(), target_column="target")


# ─────────────────────────────────────────────
# kfold_split
# ─────────────────────────────────────────────


def test_kfold_returns_correct_fold_count(sample_df) -> None:
    splits = kfold_split(sample_df, target_column="target", n_splits=5)
    assert len(splits) == 5


def test_kfold_each_split_is_tuple(sample_df) -> None:
    splits = kfold_split(sample_df, target_column="target", n_splits=4)
    for split in splits:
        assert len(split) == 4


def test_kfold_target_excluded(sample_df) -> None:
    splits = kfold_split(sample_df, target_column="target", n_splits=4)
    for X_train, X_test, _, _ in splits:
        assert "target" not in X_train.columns
        assert "target" not in X_test.columns


def test_kfold_invalid_n_splits_raises(sample_df) -> None:
    with pytest.raises(ConfigurationError, match="n_splits"):
        kfold_split(sample_df, target_column="target", n_splits=1)


def test_kfold_missing_target_raises(sample_df) -> None:
    with pytest.raises(DataValidationError, match="Target column not found"):
        kfold_split(sample_df, target_column="missing")


def test_kfold_empty_dataframe_raises() -> None:
    with pytest.raises(DataValidationError, match="empty"):
        kfold_split(pd.DataFrame(), target_column="target")


def test_kfold_reproducible(sample_df) -> None:
    splits_1 = kfold_split(sample_df, target_column="target", n_splits=5, random_state=42)
    splits_2 = kfold_split(sample_df, target_column="target", n_splits=5, random_state=42)
    for (X1, _, _, _), (X2, _, _, _) in zip(splits_1, splits_2):
        assert X1.index.tolist() == X2.index.tolist()


# ─────────────────────────────────────────────
# timeseries_split
# ─────────────────────────────────────────────


def test_timeseries_returns_correct_fold_count(sample_df) -> None:
    splits = timeseries_split(sample_df, target_column="target", n_splits=4)
    assert len(splits) == 4


def test_timeseries_preserves_order(sample_df) -> None:
    splits = timeseries_split(sample_df, target_column="target", n_splits=4)
    for X_train, X_test, _, _ in splits:
        assert X_train.index.max() < X_test.index.min()


def test_timeseries_target_excluded(sample_df) -> None:
    splits = timeseries_split(sample_df, target_column="target", n_splits=3)
    for X_train, X_test, _, _ in splits:
        assert "target" not in X_train.columns
        assert "target" not in X_test.columns


def test_timeseries_invalid_n_splits_raises(sample_df) -> None:
    with pytest.raises(ConfigurationError, match="n_splits"):
        timeseries_split(sample_df, target_column="target", n_splits=1)


def test_timeseries_missing_target_raises(sample_df) -> None:
    with pytest.raises(DataValidationError, match="Target column not found"):
        timeseries_split(sample_df, target_column="missing")


def test_timeseries_empty_dataframe_raises() -> None:
    with pytest.raises(DataValidationError, match="empty"):
        timeseries_split(pd.DataFrame(), target_column="target")
