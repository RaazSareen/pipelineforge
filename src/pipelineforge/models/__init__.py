"""
PipelineForge Model Layer

Modules
-------
classifier.py
    Classification model registry, instantiation, and fitting.
    Supports: random_forest, logistic_regression, gradient_boosting.

regressor.py
    Regression model registry, instantiation, and fitting.
    Supports: random_forest, linear_regression, gradient_boosting, ridge.

validator.py
    Model validation helpers. Placeholder — implemented in Step 9.
"""

from pipelineforge.models.classifier import (
    build_classifier,
    get_available_classifiers,
    train_classifier,
)
from pipelineforge.models.regressor import (
    build_regressor,
    get_available_regressors,
    train_regressor,
)

__all__ = [
    "get_available_classifiers",
    "build_classifier",
    "train_classifier",
    "get_available_regressors",
    "build_regressor",
    "train_regressor",
]
