import tomllib
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[3]
CONFIG_PATH = ROOT_DIR / "configs" / "default.toml"


def load_config() -> dict:
    """Load project TOML configuration."""
    with open(CONFIG_PATH, "rb") as file:
        return tomllib.load(file)
