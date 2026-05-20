"""
PipelineForge Visualization Layer

Modules
-------
plots.py
    Distribution, correlation heatmap, feature importance,
    confusion matrix, residual plots.
    All functions return matplotlib.axes.Axes.
    Optional save_path for file output.

style.py
    Shared style constants: figsize, dpi, palette, color.
"""

from pipelineforge.visualization.plots import (
    plot_confusion_matrix,
    plot_correlation_heatmap,
    plot_distribution,
    plot_feature_importance,
    plot_residuals,
)

__all__ = [
    "plot_distribution",
    "plot_correlation_heatmap",
    "plot_feature_importance",
    "plot_confusion_matrix",
    "plot_residuals",
]
