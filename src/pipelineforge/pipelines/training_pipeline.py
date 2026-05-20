"""
Deprecated compatibility wrapper.

Use instead:
    pipelineforge.orchestration.classification_pipeline
    pipelineforge.orchestration.regression_pipeline

This module will be removed in v1.0.
"""

from __future__ import annotations

import warnings


def run_training_pipeline(*args: object, **kwargs: object) -> None:
    """
    Deprecated. Use orchestration pipelines instead.

    Raises
    ------
    NotImplementedError
    """
    warnings.warn(
        (
            "pipelineforge.pipelines.training_pipeline "
            "is deprecated and will be removed in v1.0. "
            "Use pipelineforge.orchestration instead."
        ),
        DeprecationWarning,
        stacklevel=2,
    )
    raise NotImplementedError(
        "Use pipelineforge.orchestration.classification_pipeline "
        "or pipelineforge.orchestration.regression_pipeline instead."
    )
