from pathlib import Path

from pipelineforge.core.paths import ProjectPaths


def test_project_root_exists() -> None:
    assert isinstance(ProjectPaths.root, Path)


def test_configs_directory_path() -> None:
    assert ProjectPaths.configs.name == "configs"


def test_outputs_directory_path() -> None:
    assert ProjectPaths.outputs.name == "outputs"


def test_create_project_directories() -> None:
    ProjectPaths.create_project_directories()

    assert ProjectPaths.logs.exists()
