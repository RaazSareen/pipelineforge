# Changelog

All notable changes to PipelineForge will be documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/).

---

## [0.9.0] - 2026-05-19

### Added
- Evaluation layer: `classification.py`, `regression.py`, `reporter.py`
- Classification metrics: accuracy, precision, recall, f1, roc_auc, confusion_matrix, classification_report
- Regression metrics: mae, mse, rmse, r2, mape
- Metric report formatting via `format_metrics_report` and `print_metrics_report`
- ROC AUC nan guard for single-class y_true
- Full test coverage: 47 new tests across evaluation modules

---

## [0.8.0] - 2026-05-19

### Added
- Models layer: `classifier.py`, `regressor.py`
- Classifier registry: random_forest, logistic_regression, gradient_boosting
- Regressor registry: random_forest, linear_regression, gradient_boosting, ridge
- `build_classifier`, `train_classifier`, `build_regressor`, `train_regressor`
- Accepts pre-split data — splitting delegated to model_selection layer
- Full test coverage: 27 new tests across model modules

---

## [0.7.0] - 2026-05-19

### Added
- Model selection layer: `model_selection/splitter.py`
- Splitting functions: train_test_split_data, stratified_split, kfold_split, timeseries_split
- Config-compatible: accepts parameters directly, no config loading
- Exception mapping: DataValidationError for data issues, ConfigurationError for invalid parameters
- Time-series order preservation validated in tests
- Full test coverage: 26 new tests

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