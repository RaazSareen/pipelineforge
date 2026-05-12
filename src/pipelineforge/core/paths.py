"""
PipelineForge Path Management System
Version: 0.2.0
"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]

CONFIGS_DIR = PROJECT_ROOT / "configs"

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

OUTPUTS_DIR = PROJECT_ROOT / "outputs"
FIGURES_DIR = OUTPUTS_DIR / "figures"
MODELS_DIR = OUTPUTS_DIR / "models"
REPORTS_DIR = OUTPUTS_DIR / "reports"

NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"

TESTS_DIR = PROJECT_ROOT / "tests"

DOCS_DIR = PROJECT_ROOT / "docs"


def create_project_directories() -> None:
    """
    Create required framework directories.
    """

    directories = [
        CONFIGS_DIR,
        RAW_DATA_DIR,
        PROCESSED_DATA_DIR,
        FIGURES_DIR,
        MODELS_DIR,
        REPORTS_DIR,
        NOTEBOOKS_DIR,
        TESTS_DIR,
        DOCS_DIR,
    ]

    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
