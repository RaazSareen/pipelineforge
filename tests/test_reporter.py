"""
Tests: evaluation/reporter.py
Version: 0.9.0
"""

import pytest
from pipelineforge.evaluation.reporter import format_metrics_report
from pipelineforge.exceptions import DataValidationError


def test_format_report_contains_title() -> None:
    metrics = {"accuracy": 0.95, "f1": 0.93}
    report = format_metrics_report(metrics, title="Test Report")
    assert "Test Report" in report


def test_format_report_contains_keys() -> None:
    metrics = {"accuracy": 0.95, "f1": 0.93}
    report = format_metrics_report(metrics)
    assert "ACCURACY" in report
    assert "F1" in report


def test_format_report_contains_values() -> None:
    metrics = {"accuracy": 0.9500}
    report = format_metrics_report(metrics, decimals=4)
    assert "0.9500" in report


def test_format_report_custom_decimals() -> None:
    metrics = {"mae": 0.123456}
    report = format_metrics_report(metrics, decimals=2)
    assert "0.12" in report


def test_format_report_empty_metrics_raises() -> None:
    with pytest.raises(DataValidationError, match="empty"):
        format_metrics_report({})


def test_format_report_negative_decimals_raises() -> None:
    metrics = {"accuracy": 0.9}
    with pytest.raises(DataValidationError, match="decimals"):
        format_metrics_report(metrics, decimals=-1)


def test_format_report_returns_string() -> None:
    metrics = {"accuracy": 0.9, "f1": 0.88}
    report = format_metrics_report(metrics)
    assert isinstance(report, str)


def test_format_report_separator_lines() -> None:
    metrics = {"accuracy": 0.9}
    report = format_metrics_report(metrics)
    assert "=" * 10 in report
