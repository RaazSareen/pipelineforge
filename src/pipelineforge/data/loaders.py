from pathlib import Path

import pandas as pd


def load_csv(file_path: str | Path) -> pd.DataFrame:
    """
    Load CSV dataset.

    Parameters
    ----------
    file_path : str | Path
        Dataset path.

    Returns
    -------
    pd.DataFrame
        Loaded dataframe.
    """
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    return pd.read_csv(path)
