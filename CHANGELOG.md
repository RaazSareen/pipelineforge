# Changelog

All notable changes to PipelineForge will be documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/).

---

## [0.6.0] - 2026-05-16

### Added
- Preprocessing layer: `cleaner.py`, `imputer.py`, `encoder.py`, `scaler.py`, `selector.py`
- Cleaning: drop missing rows (with threshold), drop duplicates, strip whitespace, rename columns, cast types
- Imputation: mean, median, mode, constant strategies
- Encoding: one-hot, label encoding
- Scaling: min-max, standard scaling
- Selection: drop columns, select columns, drop low variance
- Full test coverage: 58 new tests across preprocessing modules
- Pandas 2.x compatibility fixes for string dtype detection

---

## [0.5.0] - 2026-05-14

### Added
- Full data loading layer: `load_csv`, `load_excel`, `load_parquet`, `load_auto`
- Format auto-detection via extension dispatch
- Logging integration in all loaders
- Extension validation with `DataValidationError`
- `ValidationResult` typed result collector
- Granular validators: `validate_dataframe_not_empty`, `validate_required_columns`,
  `validate_no_missing_values`, `validate_column_types`, `check_duplicate_rows`
- Validator chaining via shared `ValidationResult`
- Backward-compatible `validate_dataframe()` composite validator
- Full test coverage: `test_loaders.py`, `test_validation.py`
- Cleaned up `utils/logger.py` — now delegates to `core/logging.py`

---

## [0.4.0] - 2026-05-13

### Added
- Centralized framework logging system
- File + console logging handlers
- Reusable logger factory (`setup_logger`)
- Typed configuration schema system
- TOML configuration loader
- Structured application configuration models (`AppConfig`, `DataConfig`, `TrainingConfig`)
- Configuration validation layer
- Configuration and logging unit tests

---

## [0.3.0] - 2026-05-13

### Added
- Centralized framework exception hierarchy
- Custom `PipelineForgeError` base class
- `ConfigurationError`, `DataValidationError`, `PipelineExecutionError`
- Exception inheritance tests

---

## [0.2.0] - 2026-05-13

### Added
- Centralized framework path management (`ProjectPaths`)
- Core filesystem abstraction layer
- Automatic project directory creation utility
- Path validation tests

---

## [0.1.0] - 2026-05-12

### Added
- Config-driven architecture
- Centralized TOML configuration loader
- Logging utility system
- Data validation layer
- CSV data loading utilities
- Initial training pipeline skeleton
- Unit testing structure
- Ruff + pre-commit integration
- CI workflow