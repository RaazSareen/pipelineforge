from pipelineforge.exceptions import (
    ConfigurationError,
    DataValidationError,
    PipelineExecutionError,
    PipelineForgeError,
)


def test_pipelineforge_error_inheritance() -> None:
    assert issubclass(PipelineForgeError, Exception)


def test_configuration_error_inheritance() -> None:
    assert issubclass(ConfigurationError, PipelineForgeError)


def test_data_validation_error_inheritance() -> None:
    assert issubclass(DataValidationError, PipelineForgeError)


def test_pipeline_execution_error_inheritance() -> None:
    assert issubclass(PipelineExecutionError, PipelineForgeError)
