from pipelineforge.config.schema import AppConfig
from pipelineforge.config.settings import load_config


def test_load_config() -> None:
    config = load_config("configs/default.toml")

    assert isinstance(config, AppConfig)


def test_training_model_type() -> None:
    config = load_config("configs/default.toml")

    assert config.training.model_type == "random_forest"


def test_target_column() -> None:
    config = load_config("configs/default.toml")

    assert config.data.target_column == "target"
