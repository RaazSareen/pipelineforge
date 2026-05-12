import pandas as pd
import pytest
from pipelineforge.data.validation import validate_dataframe


def test_validate_dataframe_success() -> None:
    dataframe = pd.DataFrame(
        {
            "feature": [1, 2, 3],
        }
    )

    assert validate_dataframe(
        dataframe=dataframe,
        required_columns=["feature"],
    )


def test_validate_dataframe_empty() -> None:
    dataframe = pd.DataFrame()

    with pytest.raises(ValueError):
        validate_dataframe(dataframe)


def test_validate_dataframe_missing_column() -> None:
    dataframe = pd.DataFrame(
        {
            "feature": [1, 2, 3],
        }
    )

    with pytest.raises(ValueError):
        validate_dataframe(
            dataframe=dataframe,
            required_columns=["target"],
        )
