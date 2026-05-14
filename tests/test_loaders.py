"""
Tests: loaders.py
Version: 0.5.0
"""

import pandas as pd
import pytest
from pipelineforge.data.loaders import load_auto, load_csv, load_parquet
from pipelineforge.exceptions import DataValidationError


@pytest.fixture
def sample_csv(tmp_path):
    path = tmp_path / "sample.csv"
    df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    df.to_csv(path, index=False)
    return path


@pytest.fixture
def sample_parquet(tmp_path):
    path = tmp_path / "sample.parquet"
    df = pd.DataFrame({"x": [10, 20], "y": [30, 40]})
    df.to_parquet(path, index=False)
    return path


# ─────────────────────────────────────────────
# load_csv
# ─────────────────────────────────────────────


def test_load_csv_success(sample_csv) -> None:
    df = load_csv(sample_csv)
    assert isinstance(df, pd.DataFrame)
    assert list(df.columns) == ["a", "b"]
    assert len(df) == 3


def test_load_csv_file_not_found(tmp_path) -> None:
    with pytest.raises(FileNotFoundError):
        load_csv(tmp_path / "missing.csv")


def test_load_csv_unsupported_extension(tmp_path) -> None:
    bad_file = tmp_path / "data.json"
    bad_file.write_text("{}")
    with pytest.raises(DataValidationError, match="Unsupported"):
        load_csv(bad_file)


# ─────────────────────────────────────────────
# load_parquet
# ─────────────────────────────────────────────


def test_load_parquet_success(sample_parquet) -> None:
    df = load_parquet(sample_parquet)
    assert isinstance(df, pd.DataFrame)
    assert list(df.columns) == ["x", "y"]
    assert len(df) == 2


def test_load_parquet_file_not_found(tmp_path) -> None:
    with pytest.raises(FileNotFoundError):
        load_parquet(tmp_path / "missing.parquet")


# ─────────────────────────────────────────────
# load_auto
# ─────────────────────────────────────────────


def test_load_auto_csv(sample_csv) -> None:
    df = load_auto(sample_csv)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 3


def test_load_auto_parquet(sample_parquet) -> None:
    df = load_auto(sample_parquet)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 2


def test_load_auto_unsupported_extension(tmp_path) -> None:
    bad_file = tmp_path / "data.txt"
    bad_file.write_text("hello")
    with pytest.raises(DataValidationError, match="Cannot auto-detect"):
        load_auto(bad_file)
