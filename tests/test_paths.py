from pipelineforge.core.paths import (
    CONFIGS_DIR,
    DATA_DIR,
    OUTPUTS_DIR,
    PROJECT_ROOT,
)


def test_project_root_exists() -> None:
    assert PROJECT_ROOT.exists()


def test_data_directory_path() -> None:
    assert DATA_DIR.name == "data"


def test_outputs_directory_path() -> None:
    assert OUTPUTS_DIR.name == "outputs"


def test_configs_directory_path() -> None:
    assert CONFIGS_DIR.name == "configs"
