"""
PipelineForge Orchestration Layer

Modules
-------
base.py
    PipelineResult dataclass — shared result container
    for all pipeline executions.

classification_pipeline.py
    End-to-end classification: split → build → train → evaluate.

regression_pipeline.py
    End-to-end regression: split → build → train → evaluate.
"""

from pipelineforge.orchestration.base import PipelineResult
from pipelineforge.orchestration.classification_pipeline import (
    run_classification_pipeline,
)
from pipelineforge.orchestration.regression_pipeline import run_regression_pipeline

__all__ = [
    "PipelineResult",
    "run_classification_pipeline",
    "run_regression_pipeline",
]
