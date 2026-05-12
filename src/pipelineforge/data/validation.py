import pandas as pd


def validate_columns(
    dataframe: pd.DataFrame,
    required_columns: list[str],
) -> None:

    missing = [column for column in required_columns if column not in dataframe.columns]

    if missing:
        raise ValueError(f"Missing required columns: {missing}")
