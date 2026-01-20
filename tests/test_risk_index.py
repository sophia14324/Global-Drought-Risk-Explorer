"""Tests for the risk index computation."""

from __future__ import annotations

import pandas as pd

# Import risk index utilities from src.modelling.risk_index
from src.modelling.risk_index import compute_risk_index, derive_weights, INDICATOR_COLUMNS

def test_compute_risk_index_equal_weights() -> None:
    """Risk index should scale values to 0–100 range with equal weights."""
    data = {
        "rain_deficit_z": [0.0, 1.0, 2.0],
        "ndvi_stress_z": [0.0, 1.0, 2.0],
        "soil_moisture_z": [0.0, 1.0, 2.0],
        "temp_anomaly_z": [0.0, 1.0, 2.0],
    }
    df = pd.DataFrame(data)
    idx = compute_risk_index(df)
    assert idx.iloc[0] == 0.0
    assert idx.iloc[-1] == 100.0
    assert idx.iloc[1] == 50.0

def test_derive_weights_correlation() -> None:
    """Derived weights by correlation should sum to one."""
    df = pd.DataFrame({
        "rain_deficit_z": [0.0, 1.0, 2.0],
        "ndvi_stress_z": [0.1, 0.9, 1.8],
        "soil_moisture_z": [0.0, 2.0, 4.0],
        "temp_anomaly_z": [0.0, 0.5, 1.0],
    })
    weights = derive_weights(df, method="correlation")
    assert abs(sum(weights.values()) - 1.0) < 1e-6
