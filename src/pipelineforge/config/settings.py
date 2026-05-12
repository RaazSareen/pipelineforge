import tomllib
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[3]
CONFIG_DIR = ROOT_DIR / "configs"


def load_config(config_name: str = "default.toml") -> dict:
    """
    Load TOML configuration file.

    Parameters
    ----------
    config_name : str
        Name of config file.

    Returns
    -------
    dict
        Parsed TOML configuration.
    """
    config_path = CONFIG_DIR / config_name

    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    with open(config_path, "rb") as file:
        return tomllib.load(file)
