"""
PipelineForge Data Layer

Public API for data loading and validation.
"""

from pipelineforge.data.loaders import load_auto, load_csv, load_excel, load_parquet
from pipelineforge.data.validation import (
    ValidationResult,
    check_duplicate_rows,
    validate_column_types,
    validate_dataframe_not_empty,
    validate_no_missing_values,
    validate_required_columns,
)

__all__ = [
    # Loaders
    "load_csv",
    "load_excel",
    "load_parquet",
    "load_auto",
    # Validation
    "ValidationResult",
    "validate_required_columns",
    "validate_no_missing_values",
    "validate_dataframe_not_empty",
    "validate_column_types",
    "check_duplicate_rows",
]
