"""
PipelineForge Module

Version:
    0.4.0

Updated:
    2026-05-13

Purpose:
    Typed configuration schema definitions.
"""

from dataclasses import dataclass


@dataclass(slots=True)
class DataConfig:
    """
    Data configuration schema.
    """

    raw_data_path: str
    processed_data_path: str
    target_column: str


@dataclass(slots=True)
class TrainingConfig:
    """
    Training configuration schema.
    """

    test_size: float
    random_state: int
    model_type: str


@dataclass(slots=True)
class AppConfig:
    """
    Root application configuration.
    """

    data: DataConfig
    training: TrainingConfig
