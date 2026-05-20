"""
Tests: visualization/plots.py
Version: 0.10.0
"""

import matplotlib

matplotlib.use("Agg")


import pandas as pd
import pytest
from matplotlib.axes import Axes
from pipelineforge.exceptions import DataValidationError
from pipelineforge.visualization.plots import (
    plot_confusion_matrix,
    plot_correlation_heatmap,
    plot_distribution,
    plot_feature_importance,
    plot_residuals,
)


@pytest.fixture
def sample_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "a": [1.0, 2.0, 3.0, 4.0, 5.0],
            "b": [2.0, 3.0, 4.0, 5.0, 6.0],
            "c": [5.0, 3.0, 1.0, 4.0, 2.0],
        }
    )


@pytest.fixture
def confusion_df() -> pd.DataFrame:
    return pd.DataFrame(
        [[3, 1], [2, 4]],
        index=["actual_0", "actual_1"],
        columns=["predicted_0", "predicted_1"],
    )


# ─────────────────────────────────────────────
# plot_distribution
# ─────────────────────────────────────────────


def test_plot_distribution_returns_axes(sample_df) -> None:
    ax = plot_distribution(sample_df, column="a")
    assert isinstance(ax, Axes)


def test_plot_distribution_saves_file(sample_df, tmp_path) -> None:
    save_path = tmp_path / "dist.png"
    plot_distribution(sample_df, column="a", save_path=save_path)
    assert save_path.exists()


def test_plot_distribution_missing_column_raises(sample_df) -> None:
    with pytest.raises(DataValidationError, match="not found"):
        plot_distribution(sample_df, column="z")


def test_plot_distribution_non_numeric_raises() -> None:
    df = pd.DataFrame({"name": ["Alice", "Bob", "Charlie"]})
    with pytest.raises(DataValidationError, match="numeric"):
        plot_distribution(df, column="name")


def test_plot_distribution_invalid_bins_raises(sample_df) -> None:
    with pytest.raises(DataValidationError, match="bins"):
        plot_distribution(sample_df, column="a", bins=0)


# ─────────────────────────────────────────────
# plot_correlation_heatmap
# ─────────────────────────────────────────────


def test_plot_correlation_heatmap_returns_axes(sample_df) -> None:
    ax = plot_correlation_heatmap(sample_df)
    assert isinstance(ax, Axes)


def test_plot_correlation_heatmap_saves_file(sample_df, tmp_path) -> None:
    save_path = tmp_path / "corr.png"
    plot_correlation_heatmap(sample_df, save_path=save_path)
    assert save_path.exists()


def test_plot_correlation_heatmap_invalid_method_raises(sample_df) -> None:
    with pytest.raises(DataValidationError, match="Invalid method"):
        plot_correlation_heatmap(sample_df, method="bad")


def test_plot_correlation_heatmap_too_few_columns_raises() -> None:
    df = pd.DataFrame({"a": [1.0, 2.0, 3.0]})
    with pytest.raises(DataValidationError, match="At least 2"):
        plot_correlation_heatmap(df)


# ─────────────────────────────────────────────
# plot_feature_importance
# ─────────────────────────────────────────────


def test_plot_feature_importance_returns_axes() -> None:
    names = ["f1", "f2", "f3"]
    imps = [0.5, 0.3, 0.2]
    ax = plot_feature_importance(names, imps)
    assert isinstance(ax, Axes)


def test_plot_feature_importance_saves_file(tmp_path) -> None:
    save_path = tmp_path / "fi.png"
    plot_feature_importance(["f1", "f2"], [0.7, 0.3], save_path=save_path)
    assert save_path.exists()


def test_plot_feature_importance_top_n() -> None:
    names = ["f1", "f2", "f3", "f4"]
    imps = [0.4, 0.3, 0.2, 0.1]
    ax = plot_feature_importance(names, imps, top_n=2)
    assert isinstance(ax, Axes)


def test_plot_feature_importance_empty_names_raises() -> None:
    with pytest.raises(DataValidationError, match="empty"):
        plot_feature_importance([], [])


def test_plot_feature_importance_mismatched_raises() -> None:
    with pytest.raises(DataValidationError, match="same length"):
        plot_feature_importance(["f1", "f2"], [0.5])


def test_plot_feature_importance_invalid_top_n_raises() -> None:
    with pytest.raises(DataValidationError, match="top_n"):
        plot_feature_importance(["f1"], [0.5], top_n=0)


# ─────────────────────────────────────────────
# plot_confusion_matrix
# ─────────────────────────────────────────────


def test_plot_confusion_matrix_returns_axes(confusion_df) -> None:
    ax = plot_confusion_matrix(confusion_df)
    assert isinstance(ax, Axes)


def test_plot_confusion_matrix_saves_file(confusion_df, tmp_path) -> None:
    save_path = tmp_path / "cm.png"
    plot_confusion_matrix(confusion_df, save_path=save_path)
    assert save_path.exists()


def test_plot_confusion_matrix_empty_raises() -> None:
    with pytest.raises(DataValidationError, match="empty"):
        plot_confusion_matrix(pd.DataFrame())


def test_plot_confusion_matrix_non_square_raises() -> None:
    df = pd.DataFrame([[1, 2, 3], [4, 5, 6]])
    with pytest.raises(DataValidationError, match="square"):
        plot_confusion_matrix(df)


# ─────────────────────────────────────────────
# plot_residuals
# ─────────────────────────────────────────────


def test_plot_residuals_returns_axes() -> None:
    y_true = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    y_pred = pd.Series([1.1, 2.2, 2.9, 3.8, 5.1])
    ax = plot_residuals(y_true, y_pred)
    assert isinstance(ax, Axes)


def test_plot_residuals_saves_file(tmp_path) -> None:
    save_path = tmp_path / "residuals.png"
    y_true = pd.Series([1.0, 2.0, 3.0])
    y_pred = pd.Series([1.1, 1.9, 3.1])
    plot_residuals(y_true, y_pred, save_path=save_path)
    assert save_path.exists()


def test_plot_residuals_empty_y_true_raises() -> None:
    with pytest.raises(DataValidationError, match="empty"):
        plot_residuals(
            pd.Series([], dtype=float),
            pd.Series([], dtype=float),
        )


def test_plot_residuals_mismatched_length_raises() -> None:
    with pytest.raises(DataValidationError, match="same length"):
        plot_residuals(
            pd.Series([1.0, 2.0]),
            pd.Series([1.0]),
        )
