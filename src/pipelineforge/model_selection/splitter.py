"""
PipelineForge Module

Version:
    0.7.0

Updated:
    2026-05-16

Purpose:
    Dataset splitting utilities for machine learning workflows.
    Config-compatible: accepts parameters directly.
    Does not load configuration files.

    Use DataValidationError for data issues.
    Use ConfigurationError for invalid split parameters.
"""

from __future__ import annotations

import pandas as pd
from sklearn.model_selection import KFold, TimeSeriesSplit, train_test_split

from pipelineforge.core.logging import setup_logger
from pipelineforge.exceptions import ConfigurationError, DataValidationError

logger = setup_logger(__name__)

SplitResult = tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]


def _validate_dataframe(dataframe: pd.DataFrame) -> None:
    if dataframe.empty:
        raise DataValidationError("DataFrame is empty")


def _validate_target(dataframe: pd.DataFrame, target_column: str) -> None:
    if target_column not in dataframe.columns:
        raise DataValidationError(f"Target column not found: '{target_column}'")


def _validate_test_size(test_size: float) -> None:
    if not (0.0 < test_size < 1.0):
        raise ConfigurationError(f"test_size must be between 0.0 and 1.0, got {test_size}")


def _validate_n_splits(n_splits: int) -> None:
    if n_splits < 2:
        raise ConfigurationError(f"n_splits must be >= 2, got {n_splits}")


def train_test_split_data(
    dataframe: pd.DataFrame,
    target_column: str,
    test_size: float = 0.2,
    random_state: int = 42,
) -> SplitResult:
    """
    Split DataFrame into train and test sets.

    Parameters
    ----------
    dataframe : pd.DataFrame
    target_column : str
    test_size : float
        Fraction of data for test set. Must be between 0.0 and 1.0.
    random_state : int

    Returns
    -------
    tuple of (X_train, X_test, y_train, y_test)
    """
    _validate_dataframe(dataframe)
    _validate_target(dataframe, target_column)
    _validate_test_size(test_size)

    X = dataframe.drop(columns=[target_column])
    y = dataframe[target_column]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    logger.info(
        "train_test_split_data: train=%d | test=%d",
        len(X_train),
        len(X_test),
    )

    return X_train, X_test, y_train, y_test


def stratified_split(
    dataframe: pd.DataFrame,
    target_column: str,
    test_size: float = 0.2,
    random_state: int = 42,
) -> SplitResult:
    """
    Stratified train/test split preserving class distribution.

    Parameters
    ----------
    dataframe : pd.DataFrame
    target_column : str
    test_size : float
    random_state : int

    Returns
    -------
    tuple of (X_train, X_test, y_train, y_test)
    """
    _validate_dataframe(dataframe)
    _validate_target(dataframe, target_column)
    _validate_test_size(test_size)

    X = dataframe.drop(columns=[target_column])
    y = dataframe[target_column]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    logger.info(
        "stratified_split: train=%d | test=%d",
        len(X_train),
        len(X_test),
    )

    return X_train, X_test, y_train, y_test


def kfold_split(
    dataframe: pd.DataFrame,
    target_column: str,
    n_splits: int = 5,
    shuffle: bool = True,
    random_state: int = 42,
) -> list[SplitResult]:
    """
    K-Fold cross-validation split.

    Parameters
    ----------
    dataframe : pd.DataFrame
    target_column : str
    n_splits : int
        Number of folds. Must be >= 2.
    shuffle : bool
        Whether to shuffle data before splitting.
    random_state : int
        Used only when shuffle=True.

    Returns
    -------
    list of (X_train, X_test, y_train, y_test) tuples
    """
    _validate_dataframe(dataframe)
    _validate_target(dataframe, target_column)
    _validate_n_splits(n_splits)

    X = dataframe.drop(columns=[target_column])
    y = dataframe[target_column]

    kf = KFold(
        n_splits=n_splits,
        shuffle=shuffle,
        random_state=random_state if shuffle else None,
    )

    splits = []

    for fold, (train_idx, test_idx) in enumerate(kf.split(X), start=1):
        X_train = X.iloc[train_idx]
        X_test = X.iloc[test_idx]
        y_train = y.iloc[train_idx]
        y_test = y.iloc[test_idx]

        splits.append((X_train, X_test, y_train, y_test))

        logger.info(
            "kfold_split: fold=%d | train=%d | test=%d",
            fold,
            len(X_train),
            len(X_test),
        )

    return splits


def timeseries_split(
    dataframe: pd.DataFrame,
    target_column: str,
    n_splits: int = 5,
) -> list[SplitResult]:
    """
    Time-series aware cross-validation split.

    Preserves temporal order — no shuffling.
    Each fold uses all past data as training, next window as test.

    Parameters
    ----------
    dataframe : pd.DataFrame
        Must be sorted by time before passing.
    target_column : str
    n_splits : int
        Number of folds. Must be >= 2.

    Returns
    -------
    list of (X_train, X_test, y_train, y_test) tuples
    """
    _validate_dataframe(dataframe)
    _validate_target(dataframe, target_column)
    _validate_n_splits(n_splits)

    X = dataframe.drop(columns=[target_column])
    y = dataframe[target_column]

    tss = TimeSeriesSplit(n_splits=n_splits)

    splits = []

    for fold, (train_idx, test_idx) in enumerate(tss.split(X), start=1):
        X_train = X.iloc[train_idx]
        X_test = X.iloc[test_idx]
        y_train = y.iloc[train_idx]
        y_test = y.iloc[test_idx]

        splits.append((X_train, X_test, y_train, y_test))

        logger.info(
            "timeseries_split: fold=%d | train=%d | test=%d",
            fold,
            len(X_train),
            len(X_test),
        )

    return splits
