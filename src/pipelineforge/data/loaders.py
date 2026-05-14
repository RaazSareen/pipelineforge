"""
PipelineForge Module

Version:
    0.5.0

Updated:
    2026-05-14

Purpose:
    Data loading utilities with logging, validation, and format dispatch.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from pipelineforge.core.logging import setup_logger
from pipelineforge.exceptions import DataValidationError

logger = setup_logger(__name__)

_SUPPORTED_EXTENSIONS = {".csv", ".xlsx", ".xls", ".parquet"}


def _resolve_path(file_path: str | Path) -> Path:
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    if path.suffix.lower() not in _SUPPORTED_EXTENSIONS:
        raise DataValidationError(
            f"Unsupported file format: '{path.suffix}'. Supported: {sorted(_SUPPORTED_EXTENSIONS)}"
        )

    return path


def load_csv(file_path: str | Path, **kwargs) -> pd.DataFrame:
    """
    Load a CSV file into a DataFrame.

    Parameters
    ----------
    file_path : str | Path
        Path to the CSV file.
    **kwargs
        Additional arguments passed to pd.read_csv.

    Returns
    -------
    pd.DataFrame
    """
    path = _resolve_path(file_path)

    logger.info("Loading CSV: %s", path)

    df = pd.read_csv(path, **kwargs)

    logger.info("Loaded %d rows x %d cols from %s", len(df), len(df.columns), path.name)

    return df


def load_excel(file_path: str | Path, sheet_name: str | int = 0, **kwargs) -> pd.DataFrame:
    """
    Load an Excel file into a DataFrame.

    Parameters
    ----------
    file_path : str | Path
        Path to the Excel file (.xlsx or .xls).
    sheet_name : str | int
        Sheet to load. Default is first sheet.
    **kwargs
        Additional arguments passed to pd.read_excel.

    Returns
    -------
    pd.DataFrame
    """
    path = _resolve_path(file_path)

    logger.info("Loading Excel: %s | sheet=%s", path, sheet_name)

    df = pd.read_excel(path, sheet_name=sheet_name, **kwargs)

    logger.info("Loaded %d rows x %d cols from %s", len(df), len(df.columns), path.name)

    return df


def load_parquet(file_path: str | Path, **kwargs) -> pd.DataFrame:
    """
    Load a Parquet file into a DataFrame.

    Parameters
    ----------
    file_path : str | Path
        Path to the Parquet file.
    **kwargs
        Additional arguments passed to pd.read_parquet.

    Returns
    -------
    pd.DataFrame
    """
    path = _resolve_path(file_path)

    logger.info("Loading Parquet: %s", path)

    df = pd.read_parquet(path, **kwargs)

    logger.info("Loaded %d rows x %d cols from %s", len(df), len(df.columns), path.name)

    return df


def load_auto(file_path: str | Path, **kwargs) -> pd.DataFrame:
    """
    Auto-detect file format from extension and load.

    Supports: .csv, .xlsx, .xls, .parquet

    Parameters
    ----------
    file_path : str | Path
        Path to the data file.
    **kwargs
        Additional arguments forwarded to the format-specific loader.

    Returns
    -------
    pd.DataFrame
    """
    path = Path(file_path)
    ext = path.suffix.lower()

    dispatch = {
        ".csv": load_csv,
        ".xlsx": load_excel,
        ".xls": load_excel,
        ".parquet": load_parquet,
    }

    if ext not in dispatch:
        raise DataValidationError(
            f"Cannot auto-detect loader for extension: '{ext}'. "
            f"Supported: {sorted(dispatch.keys())}"
        )

    logger.info("Auto-detected format '%s' for: %s", ext, path.name)

    return dispatch[ext](file_path, **kwargs)
