"""
PipelineForge Feature Engineering Layer

Modules
-------
generator.py
    Derived feature creation: ratios, differences, row-wise aggregates.
    Input-driven only — no statistical tests, no model outputs.

transformer.py
    Mathematical and structural transformations: log, polynomial,
    binning, datetime decomposition.

interaction.py
    Pairwise and multiplicative feature interactions.

selector.py
    Statistical and model-informed feature selection: correlation filters,
    VIF elimination, importance-based selection from fitted estimators.
    Complements preprocessing/selector.py which handles structural selection.
"""

from pipelineforge.features.generator import (
    add_aggregate_feature,
    add_difference_feature,
    add_ratio_feature,
)
from pipelineforge.features.interaction import (
    add_division_interaction,
    add_interaction_features,
    add_polynomial_interaction,
)
from pipelineforge.features.transformer import (
    apply_binning,
    apply_log_transform,
    apply_polynomial_features,
    extract_datetime_features,
)

__all__ = [
    "add_ratio_feature",
    "add_difference_feature",
    "add_aggregate_feature",
    "apply_log_transform",
    "apply_polynomial_features",
    "apply_binning",
    "extract_datetime_features",
    "add_interaction_features",
    "add_division_interaction",
    "add_polynomial_interaction",
]
