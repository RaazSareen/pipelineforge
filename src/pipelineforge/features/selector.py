"""
PipelineForge Module

Version:
    0.7.0

Updated:
    2026-05-16

Purpose:
    Statistical and model-informed feature selection.
    Includes correlation filters, VIF elimination, and importance-based
    selection from fitted estimators.

    Complements preprocessing/selector.py which handles structural selection
    (drop by name, select by name, low variance). This module operates on
    statistical relationships between features and targets.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator

from pipelineforge.core.logging import setup_logger
from pipelineforge.exceptions import DataValidationError

logger = setup_logger(__name__)


def drop_highly_correlated_features(
    dataframe: pd.DataFrame,
    threshold: float = 0.9,
    method: str = "pearson",
) -> pd.DataFrame:
    """
    Drop numeric features whose pairwise correlation exceeds the threshold.

    Iterates upper triangle of the correlation matrix. For each correlated
    pair, the second column encountered is dropped.

    Parameters
    ----------
    dataframe : pd.DataFrame
    threshold : float
        Absolute correlation value above which a feature is dropped.
        Must be between 0.0 and 1.0.
    method : str
        Correlation method: 'pearson', 'spearman', or 'kendall'.

    Returns
    -------
    pd.DataFrame
    """
    valid_methods = {"pearson", "spearman", "kendall"}

    if method not in valid_methods:
        raise DataValidationError(
            f"Invalid method: '{method}'. Must be one of {sorted(valid_methods)}"
        )

    if not (0.0 < threshold <= 1.0):
        raise DataValidationError(f"threshold must be between 0.0 and 1.0, got {threshold}")

    numeric_df = dataframe.select_dtypes(include="number")

    if numeric_df.shape[1] < 2:
        raise DataValidationError(
            "At least 2 numeric columns are required for correlation analysis"
        )

    corr_matrix = numeric_df.corr(method=method).abs()

    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

    to_drop = [col for col in upper.columns if any(upper[col] > threshold)]

    if to_drop:
        logger.info(
            "drop_highly_correlated_features: dropping %d column(s): %s",
            len(to_drop),
            to_drop,
        )

    return dataframe.drop(columns=to_drop)


def select_by_target_correlation(
    dataframe: pd.DataFrame,
    target_column: str,
    threshold: float = 0.1,
    method: str = "pearson",
) -> pd.DataFrame:
    """
    Keep only features whose absolute correlation with the target
    meets or exceeds the threshold.

    The target column itself is always retained.

    Parameters
    ----------
    dataframe : pd.DataFrame
    target_column : str
        Name of the target column.
    threshold : float
        Minimum absolute correlation required to keep a feature.
    method : str
        Correlation method: 'pearson', 'spearman', or 'kendall'.

    Returns
    -------
    pd.DataFrame
    """
    valid_methods = {"pearson", "spearman", "kendall"}

    if method not in valid_methods:
        raise DataValidationError(
            f"Invalid method: '{method}'. Must be one of {sorted(valid_methods)}"
        )

    if target_column not in dataframe.columns:
        raise DataValidationError(f"Target column not found: '{target_column}'")

    if not (0.0 <= threshold <= 1.0):
        raise DataValidationError(f"threshold must be between 0.0 and 1.0, got {threshold}")

    numeric_df = dataframe.select_dtypes(include="number")

    correlations = numeric_df.corr(method=method)[target_column].abs()

    selected = correlations[correlations >= threshold].index.tolist()

    non_numeric = [
        col for col in dataframe.columns if col not in numeric_df.columns and col != target_column
    ]

    final_cols = list(dict.fromkeys(selected + non_numeric))

    if target_column not in final_cols:
        final_cols.append(target_column)

    logger.info(
        "select_by_target_correlation: kept %d feature(s) with threshold=%.2f",
        len(final_cols),
        threshold,
    )

    return dataframe[final_cols]


def calculate_vif(
    dataframe: pd.DataFrame,
    columns: list[str] | None = None,
) -> pd.DataFrame:
    """
    Calculate Variance Inflation Factor (VIF) for numeric columns.

    Requires statsmodels (optional extra: uv sync --extra stats).

    Parameters
    ----------
    dataframe : pd.DataFrame
    columns : list[str] | None
        Columns to evaluate. Auto-selects numeric columns if None.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns ['feature', 'vif'], sorted by VIF descending.
    """
    try:
        from statsmodels.stats.outliers_influence import variance_inflation_factor
    except ImportError as exc:
        raise ImportError(
            "statsmodels is required for VIF calculation. Install with: uv sync --extra stats"
        ) from exc

    target_cols = (
        columns
        if columns is not None
        else dataframe.select_dtypes(include="number").columns.tolist()
    )

    missing = [col for col in target_cols if col not in dataframe.columns]
    if missing:
        raise DataValidationError(f"Columns not found: {missing}")

    if len(target_cols) < 2:
        raise DataValidationError("At least 2 columns are required for VIF calculation")

    subset = dataframe[target_cols].dropna()

    vif_data = pd.DataFrame(
        {
            "feature": target_cols,
            "vif": [variance_inflation_factor(subset.values, i) for i in range(len(target_cols))],
        }
    )

    return vif_data.sort_values("vif", ascending=False).reset_index(drop=True)


def drop_high_vif_features(
    dataframe: pd.DataFrame,
    threshold: float = 5.0,
    columns: list[str] | None = None,
) -> pd.DataFrame:
    """
    Iteratively drop the highest-VIF feature until all VIF values
    are at or below the threshold.

    Requires statsmodels (optional extra: uv sync --extra stats).

    Parameters
    ----------
    dataframe : pd.DataFrame
    threshold : float
        Maximum acceptable VIF. Default 5.0.
    columns : list[str] | None
        Columns to evaluate. Auto-selects numeric columns if None.

    Returns
    -------
    pd.DataFrame
    """
    if threshold <= 0:
        raise DataValidationError(f"threshold must be > 0, got {threshold}")

    result = dataframe.copy()

    target_cols = (
        list(columns)
        if columns is not None
        else result.select_dtypes(include="number").columns.tolist()
    )

    while True:
        if len(target_cols) < 2:
            break

        vif_df = calculate_vif(result, columns=target_cols)
        max_vif = vif_df["vif"].iloc[0]

        if max_vif <= threshold:
            break

        drop_col = vif_df["feature"].iloc[0]

        logger.info(
            "drop_high_vif_features: dropping '%s' with VIF=%.2f",
            drop_col,
            max_vif,
        )

        result = result.drop(columns=[drop_col])
        target_cols.remove(drop_col)

    return result


def select_by_feature_importance(
    dataframe: pd.DataFrame,
    estimator: BaseEstimator,
    target_column: str,
    threshold: float = 0.0,
) -> pd.DataFrame:
    """
    Select features using importance scores from an already-fitted estimator.

    Reads feature_importances_ (tree-based) or coef_ (linear models).
    Does not train any model — accepts only a fitted estimator.

    Parameters
    ----------
    dataframe : pd.DataFrame
    estimator : BaseEstimator
        A fitted sklearn estimator with feature_importances_ or coef_.
    target_column : str
        Target column name — excluded from features, always retained.
    threshold : float
        Minimum importance score to keep a feature. Default 0.0.

    Returns
    -------
    pd.DataFrame
        DataFrame containing only selected features and the target column.
    """
    if target_column not in dataframe.columns:
        raise DataValidationError(f"Target column not found: '{target_column}'")

    if threshold < 0:
        raise DataValidationError(f"threshold must be >= 0, got {threshold}")

    feature_cols = [col for col in dataframe.columns if col != target_column]

    if hasattr(estimator, "feature_importances_"):
        importances = estimator.feature_importances_
    elif hasattr(estimator, "coef_"):
        coef = estimator.coef_
        if coef.ndim > 1:
            importances = abs(coef).mean(axis=0)
        else:
            importances = abs(coef)
    else:
        raise DataValidationError(
            "Estimator must have feature_importances_ or coef_ attribute. "
            "Ensure the estimator is fitted before passing it."
        )

    if len(importances) != len(feature_cols):
        raise DataValidationError(
            f"Estimator has {len(importances)} feature(s) but DataFrame "
            f"has {len(feature_cols)} feature column(s). "
            "Ensure the estimator was trained on the same feature set."
        )

    selected = [col for col, imp in zip(feature_cols, importances) if imp > threshold]

    if not selected:
        raise DataValidationError(
            f"No features met the importance threshold of {threshold}. "
            "Consider lowering the threshold."
        )

    logger.info(
        "select_by_feature_importance: selected %d / %d feature(s) with threshold=%.4f",
        len(selected),
        len(feature_cols),
        threshold,
    )

    return dataframe[selected + [target_column]]
