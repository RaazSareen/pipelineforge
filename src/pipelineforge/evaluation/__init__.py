"""
PipelineForge Evaluation Layer

Modules
-------
classification.py
    Classification metrics: accuracy, precision, recall, f1,
    roc_auc, confusion_matrix, classification_report.
    Accepts y_true and y_pred directly.

regression.py
    Regression metrics: mae, mse, rmse, r2, mape.
    Accepts y_true and y_pred directly.

reporter.py
    Metric report formatting and display.
    Does not compute metrics.

validator.py
    Evaluation validation helpers. Placeholder — Step 11.
"""

from pipelineforge.evaluation.classification import (
    compute_accuracy,
    compute_all_classification_metrics,
    compute_classification_report,
    compute_confusion_matrix,
    compute_f1,
    compute_precision,
    compute_recall,
    compute_roc_auc,
)
from pipelineforge.evaluation.regression import (
    compute_all_regression_metrics,
    compute_mae,
    compute_mape,
    compute_mse,
    compute_r2,
    compute_rmse,
)
from pipelineforge.evaluation.reporter import format_metrics_report, print_metrics_report

__all__ = [
    "compute_accuracy",
    "compute_precision",
    "compute_recall",
    "compute_f1",
    "compute_roc_auc",
    "compute_confusion_matrix",
    "compute_classification_report",
    "compute_all_classification_metrics",
    "compute_mae",
    "compute_mse",
    "compute_rmse",
    "compute_r2",
    "compute_mape",
    "compute_all_regression_metrics",
    "format_metrics_report",
    "print_metrics_report",
]
