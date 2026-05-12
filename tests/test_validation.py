import pandas as pd
import pytest
from pipelineforge.data.validation import validate_columns


def test_validate_columns_success() -> None:
    dataframe = pd.DataFrame(
        {
            "age": [20],
            "salary": [1000],
        }
    )

    validate_columns(
        dataframe=dataframe,
        required_columns=["age", "salary"],
    )


def test_validate_columns_failure() -> None:
    dataframe = pd.DataFrame(
        {
            "age": [20],
        }
    )

    with pytest.raises(ValueError):
        validate_columns(
            dataframe=dataframe,
            required_columns=["age", "salary"],
        )
