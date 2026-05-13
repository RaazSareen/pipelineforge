"""
PipelineForge Configuration Loader
Version: 0.4.0
"""

import tomllib
from pathlib import Path

from pipelineforge.config.schema import (
    AppConfig,
    DataConfig,
    TrainingConfig,
)
from pipelineforge.exceptions import ConfigurationError


def load_config(config_path: str | Path) -> AppConfig:
    """
    Load and validate TOML configuration.
    """

    path = Path(config_path)

    if not path.exists():
        raise ConfigurationError(f"Configuration file not found: {path}")

    with path.open("rb") as file:
        config_data = tomllib.load(file)

    try:
        data_config = DataConfig(
            raw_data_path=config_data["data"]["raw_data_path"],
            processed_data_path=config_data["data"]["processed_data_path"],
            target_column=config_data["data"]["target_column"],
        )

        training_config = TrainingConfig(
            test_size=config_data["training"]["test_size"],
            random_state=config_data["training"]["random_state"],
            model_type=config_data["training"]["model_type"],
        )

        return AppConfig(
            data=data_config,
            training=training_config,
        )

    except KeyError as error:
        raise ConfigurationError(f"Missing configuration key: {error}") from error
