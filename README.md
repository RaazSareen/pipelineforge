# PipelineForge

A modular, reproducible, production-grade Data Science engineering framework designed for scalable end-to-end machine learning workflows.

## Core Features

- Modular project architecture
- Reproducible environments via `uv`
- Config-driven execution
- Data ingestion and preprocessing pipelines
- EDA and visualization workflows
- Feature engineering utilities
- Model training and evaluation
- Ruff linting and formatting
- Pytest-based testing
- GitHub Actions CI/CD

## Tech Stack

Python 3.12+ · NumPy · Pandas · Scikit-learn · Matplotlib · uv · Ruff · pytest · GitHub Actions

## Project Structure

```text
pipelineforge/
├── configs/          # Config files (YAML, TOML)
├── pipelines/        # End-to-end pipeline scripts
├── src/pipelineforge/# Core Python package
├── tests/            # Unit and integration tests
├── data/             # raw/ and processed/
├── notebooks/        # EDA and exploration
├── outputs/          # figures/ models/ reports/
├── docs/             # Documentation
└── scripts/          # Utility scripts
```

## Setup

```powershell
uv venv
uv sync --extra dev
```

## Development

```powershell
uv run ruff check .
uv run pytest
```
## Versioning

PipelineForge follows Semantic Versioning (SemVer).

Current Version: `0.2.0`