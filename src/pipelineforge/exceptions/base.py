"""
PipelineForge Module

Version:
    0.3.0

Updated:
    2026-05-13

Purpose:
    Base framework exception hierarchy.
"""


class PipelineForgeError(Exception):
    """
    Base exception for all PipelineForge errors.
    """


class ConfigurationError(PipelineForgeError):
    """
    Raised when configuration is invalid.
    """


class DataValidationError(PipelineForgeError):
    """
    Raised when data validation fails.
    """


class PipelineExecutionError(PipelineForgeError):
    """
    Raised when pipeline execution fails.
    """
