from logging import Logger

from pipelineforge.core.logging import setup_logger


def test_setup_logger_returns_logger() -> None:
    logger = setup_logger("test_logger")

    assert isinstance(logger, Logger)


def test_logger_name() -> None:
    logger = setup_logger("pipelineforge.test")

    assert logger.name == "pipelineforge.test"


def test_logger_has_handlers() -> None:
    logger = setup_logger("handlers_test")

    assert len(logger.handlers) >= 2
