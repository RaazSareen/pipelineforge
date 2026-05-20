"""
PipelineForge Module

Version:
    0.10.0

Updated:
    2026-05-19

Purpose:
    Visualization utilities for data exploration and model evaluation.
    All functions return matplotlib.axes.Axes for composability.
    Optionally save to file via save_path.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes

from pipelineforge.core.logging import setup_logger
from pipelineforge.exceptions import DataValidationError
from pipelineforge.visualization.style import (
    DEFAULT_COLOR,
    DEFAULT_DPI,
    DEFAULT_FIGSIZE,
    DEFAULT_PALETTE,
)

logger = setup_logger(__name__)


def _save_figure(fig: plt.Figure, save_path: str | Path) -> None:
    path = Path(save_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=DEFAULT_DPI, bbox_inches="tight")
    logger.info("Figure saved: %s", path)


def plot_distribution(
    dataframe: pd.DataFrame,
    column: str,
    bins: int = 30,
    figsize: tuple[int, int] = DEFAULT_FIGSIZE,
    save_path: str | Path | None = None,
) -> Axes:
    """
    Plot histogram distribution of a numeric column.

    Parameters
    ----------
    dataframe : pd.DataFrame
    column : str
        Numeric column to plot.
    bins : int
        Number of histogram bins.
    figsize : tuple[int, int]
    save_path : str | Path | None
        If provided, saves figure to this path.

    Returns
    -------
    matplotlib.axes.Axes
    """
    if column not in dataframe.columns:
        raise DataValidationError(f"Column not found: '{column}'")

    if not pd.api.types.is_numeric_dtype(dataframe[column]):
        raise DataValidationError(f"Column '{column}' must be numeric for distribution plot")

    if bins < 1:
        raise DataValidationError(f"bins must be >= 1, got {bins}")

    fig, ax = plt.subplots(figsize=figsize)

    ax.hist(dataframe[column].dropna(), bins=bins, color=DEFAULT_COLOR, edgecolor="white")
    ax.set_title(f"Distribution of {column}")
    ax.set_xlabel(column)
    ax.set_ylabel("Frequency")

    if save_path:
        _save_figure(fig, save_path)

    plt.close(fig)

    logger.info("plot_distribution: '%s'", column)

    return ax


def plot_correlation_heatmap(
    dataframe: pd.DataFrame,
    method: str = "pearson",
    figsize: tuple[int, int] = DEFAULT_FIGSIZE,
    save_path: str | Path | None = None,
) -> Axes:
    """
    Plot correlation heatmap for numeric columns.

    Parameters
    ----------
    dataframe : pd.DataFrame
    method : str
        Correlation method: 'pearson', 'spearman', or 'kendall'.
    figsize : tuple[int, int]
    save_path : str | Path | None

    Returns
    -------
    matplotlib.axes.Axes
    """
    valid_methods = {"pearson", "spearman", "kendall"}

    if method not in valid_methods:
        raise DataValidationError(
            f"Invalid method: '{method}'. Must be one of {sorted(valid_methods)}"
        )

    numeric_df = dataframe.select_dtypes(include="number")

    if numeric_df.shape[1] < 2:
        raise DataValidationError("At least 2 numeric columns required for correlation heatmap")

    corr = numeric_df.corr(method=method)

    fig, ax = plt.subplots(figsize=figsize)

    cax = ax.imshow(corr.values, cmap=DEFAULT_PALETTE, vmin=-1, vmax=1, aspect="auto")
    fig.colorbar(cax, ax=ax)

    ticks = range(len(corr.columns))
    ax.set_xticks(list(ticks))
    ax.set_yticks(list(ticks))
    ax.set_xticklabels(corr.columns, rotation=45, ha="right")
    ax.set_yticklabels(corr.columns)
    ax.set_title(f"Correlation Heatmap ({method})")

    if save_path:
        _save_figure(fig, save_path)

    plt.close(fig)

    logger.info("plot_correlation_heatmap: method='%s'", method)

    return ax


def plot_feature_importance(
    feature_names: list[str],
    importances: list[float] | np.ndarray,
    top_n: int | None = None,
    figsize: tuple[int, int] = DEFAULT_FIGSIZE,
    save_path: str | Path | None = None,
) -> Axes:
    """
    Plot horizontal bar chart of feature importances.

    Parameters
    ----------
    feature_names : list[str]
    importances : list[float] | np.ndarray
    top_n : int | None
        If set, show only top N features by importance.
    figsize : tuple[int, int]
    save_path : str | Path | None

    Returns
    -------
    matplotlib.axes.Axes
    """
    if len(feature_names) == 0:
        raise DataValidationError("feature_names is empty")

    if len(importances) == 0:
        raise DataValidationError("importances is empty")

    if len(feature_names) != len(importances):
        raise DataValidationError(
            f"feature_names and importances must have the same length. "
            f"Got {len(feature_names)} and {len(importances)}"
        )

    if top_n is not None and top_n < 1:
        raise DataValidationError(f"top_n must be >= 1, got {top_n}")

    imp_array = np.asarray(importances)
    sorted_idx = np.argsort(imp_array)

    if top_n is not None:
        sorted_idx = sorted_idx[-top_n:]

    sorted_names = [feature_names[i] for i in sorted_idx]
    sorted_imps = imp_array[sorted_idx]

    fig, ax = plt.subplots(figsize=figsize)

    ax.barh(sorted_names, sorted_imps, color=DEFAULT_COLOR)
    ax.set_title("Feature Importance")
    ax.set_xlabel("Importance")
    ax.set_ylabel("Feature")

    if save_path:
        _save_figure(fig, save_path)

    plt.close(fig)

    logger.info("plot_feature_importance: %d features", len(sorted_names))

    return ax


def plot_confusion_matrix(
    confusion_df: pd.DataFrame,
    figsize: tuple[int, int] = DEFAULT_FIGSIZE,
    save_path: str | Path | None = None,
) -> Axes:
    """
    Plot confusion matrix from a labeled DataFrame.

    Accepts output of evaluation.classification.compute_confusion_matrix().

    Parameters
    ----------
    confusion_df : pd.DataFrame
        Square DataFrame with actual/predicted labels.
    figsize : tuple[int, int]
    save_path : str | Path | None

    Returns
    -------
    matplotlib.axes.Axes
    """
    if confusion_df.empty:
        raise DataValidationError("confusion_df is empty")

    if confusion_df.shape[0] != confusion_df.shape[1]:
        raise DataValidationError(f"confusion_df must be square. Got shape {confusion_df.shape}")

    fig, ax = plt.subplots(figsize=figsize)

    cax = ax.imshow(confusion_df.values, cmap=DEFAULT_PALETTE, aspect="auto")
    fig.colorbar(cax, ax=ax)

    ticks = range(len(confusion_df.columns))
    ax.set_xticks(list(ticks))
    ax.set_yticks(list(ticks))
    ax.set_xticklabels(confusion_df.columns, rotation=45, ha="right")
    ax.set_yticklabels(confusion_df.index)
    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")

    for i in range(confusion_df.shape[0]):
        for j in range(confusion_df.shape[1]):
            ax.text(
                j,
                i,
                str(confusion_df.values[i, j]),
                ha="center",
                va="center",
                color="black",
            )

    if save_path:
        _save_figure(fig, save_path)

    plt.close(fig)

    logger.info("plot_confusion_matrix: shape=%s", confusion_df.shape)

    return ax


def plot_residuals(
    y_true: pd.Series | np.ndarray,
    y_pred: pd.Series | np.ndarray,
    figsize: tuple[int, int] = DEFAULT_FIGSIZE,
    save_path: str | Path | None = None,
) -> Axes:
    """
    Plot residuals (y_true - y_pred) against predicted values.

    Parameters
    ----------
    y_true : pd.Series | np.ndarray
    y_pred : pd.Series | np.ndarray
    figsize : tuple[int, int]
    save_path : str | Path | None

    Returns
    -------
    matplotlib.axes.Axes
    """
    if len(y_true) == 0:
        raise DataValidationError("y_true is empty")

    if len(y_pred) == 0:
        raise DataValidationError("y_pred is empty")

    if len(y_true) != len(y_pred):
        raise DataValidationError(
            f"y_true and y_pred must have the same length. Got {len(y_true)} and {len(y_pred)}"
        )

    residuals = np.asarray(y_true) - np.asarray(y_pred)

    fig, ax = plt.subplots(figsize=figsize)

    ax.scatter(y_pred, residuals, color=DEFAULT_COLOR, alpha=0.6, edgecolors="white")
    ax.axhline(0, color="red", linewidth=1.0, linestyle="--")
    ax.set_title("Residual Plot")
    ax.set_xlabel("Predicted Values")
    ax.set_ylabel("Residuals")

    if save_path:
        _save_figure(fig, save_path)

    plt.close(fig)

    logger.info("plot_residuals: %d points", len(residuals))

    return ax
