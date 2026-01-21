"""Unit tests for the data preprocessing module."""

from __future__ import annotations

import pandas as pd

# Import the preprocessing function from src.data.preprocess
from src.data.preprocess import _monthly_zscore

def test_monthly_zscore_constant() -> None:
    """Z-score should be zero when values are constant within each group."""
    df = pd.DataFrame({
        "ADM0_NAME": ["A"] * 6,
        "ADM1_NAME": ["B"] * 6,
        "date": pd.to_datetime([
            "2020-01-15", "2020-01-20",
            "2020-02-10", "2020-02-15",
            "2020-03-01", "2020-03-20",
        ]),
        "value": [10, 10, 5, 5, 0, 0],
    })
    result = _monthly_zscore(df, "value")
    assert (result["z"] == 0.0).all()
