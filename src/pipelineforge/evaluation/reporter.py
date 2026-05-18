"""
PipelineForge Module

Version:
    0.9.0

Updated:
    2026-05-16

Purpose:
    Metric report formatting and display utilities.
    Formats metric dicts into readable string reports.
    Does not compute metrics — use classification.py or regression.py.
"""

from __future__ import annotations

from pipelineforge.core.logging import setup_logger
from pipelineforge.exceptions import DataValidationError

logger = setup_logger(__name__)


def format_metrics_report(
    metrics: dict[str, float],
    title: str = "Evaluation Report",
    decimals: int = 4,
) -> str:
    """
    Format a metrics dict into a readable report string.

    Parameters
    ----------
    metrics : dict[str, float]
        Metric name → value pairs.
    title : str
        Report header title.
    decimals : int
        Number of decimal places. Must be >= 0.

    Returns
    -------
    str
    """
    if not metrics:
        raise DataValidationError("metrics dict is empty")

    if decimals < 0:
        raise DataValidationError(f"decimals must be >= 0, got {decimals}")

    width = 40
    lines = [
        "=" * width,
        title.center(width),
        "=" * width,
    ]

    for key, value in metrics.items():
        label = key.upper().replace("_", " ")
        lines.append(f"  {label:<20} {value:.{decimals}f}")

    lines.append("=" * width)

    report = "\n".join(lines)

    logger.info("format_metrics_report: formatted %d metric(s)", len(metrics))

    return report


def print_metrics_report(
    metrics: dict[str, float],
    title: str = "Evaluation Report",
    decimals: int = 4,
) -> None:
    """
    Print a formatted metrics report to stdout.

    Parameters
    ----------
    metrics : dict[str, float]
    title : str
    decimals : int
    """
    report = format_metrics_report(metrics, title=title, decimals=decimals)
    print(report)
