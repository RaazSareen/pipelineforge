"""
PipelineForge Module

Version:
    0.4.0

Updated:
    2026-05-13

Purpose:
    Centralized project path management.
"""

from __future__ import annotations

from pathlib import Path


class ProjectPaths:
    """
    Centralized framework path registry.
    """

    root = Path(__file__).resolve().parents[3]

    configs = root / "configs"

    data = root / "data"
    raw_data = data / "raw"
    processed_data = data / "processed"

    outputs = root / "outputs"
    figures = outputs / "figures"
    models = outputs / "models"
    reports = outputs / "reports"

    notebooks = root / "notebooks"

    tests = root / "tests"

    docs = root / "docs"

    logs = root / "logs"

    @classmethod
    def create_project_directories(cls) -> None:
        """
        Create required framework directories.
        """

        directories = [
            cls.configs,
            cls.raw_data,
            cls.processed_data,
            cls.figures,
            cls.models,
            cls.reports,
            cls.notebooks,
            cls.tests,
            cls.docs,
            cls.logs,
        ]

        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
