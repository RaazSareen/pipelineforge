"""
PipelineForge Model Selection Layer

Modules
-------
splitter.py
    Train/test splitting, stratified splitting, K-Fold,
    and time-series cross-validation splits.
    Config-compatible: accepts parameters directly.

validator.py
    Cross-validation helpers: GridSearch, RandomSearch,
    nested CV. Placeholder — implemented in Step 9.
"""

from pipelineforge.model_selection.splitter import (
    kfold_split,
    stratified_split,
    timeseries_split,
    train_test_split_data,
)

__all__ = [
    "train_test_split_data",
    "stratified_split",
    "kfold_split",
    "timeseries_split",
]
