"""
PipelineForge Module

Version:
    0.4.0

Updated:
    2026-05-13

Purpose:
    Centralized framework logging utilities.
"""

from __future__ import annotations

from logging import FileHandler, Formatter, Logger, StreamHandler, getLogger
from pathlib import Path

from pipelineforge.core.paths import ProjectPaths

LOG_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"

DEFAULT_LOG_LEVEL = "INFO"


def setup_logger(
    name: str,
    log_file: str = "pipelineforge.log",
    level: str = DEFAULT_LOG_LEVEL,
) -> Logger:
    """
    Configure and return a reusable framework logger.
    """

    logger = getLogger(name)

    if logger.handlers:
        return logger

    logger.setLevel(level)

    logger.propagate = False

    formatter = Formatter(LOG_FORMAT)

    log_path: Path = ProjectPaths.root / "logs" / log_file
    log_path.parent.mkdir(parents=True, exist_ok=True)

    file_handler = FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)

    console_handler = StreamHandler()
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger
