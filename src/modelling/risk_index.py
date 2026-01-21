"""
Risk index computation utilities for the Global Drought Risk Explorer.

This module defines helper functions to derive indicator weights and
compute a composite drought risk index from multiple standardised
indicators.  It also retains a ``compute()`` function for backwards
compatibility with the original xarray-based workflow that writes a
NetCDF file.  The preferred API for downstream code is
``compute_risk_index``.
"""

from __future__ import annotations

from typing import Dict, Iterable, Optional

import numpy as np
import pandas as pd

# Names of the indicator columns expected by compute_risk_index.  These
# correspond to standardised (z-score) versions of the raw measures.
INDICATOR_COLUMNS: Iterable[str] = [
    "rain_deficit_z",
    "ndvi_stress_z",
    "soil_moisture_z",
    "temp_anomaly_z",
]

def derive_weights(
    df: pd.DataFrame,
    indicators: Iterable[str] = INDICATOR_COLUMNS,
    method: str = "correlation",
) -> Dict[str, float]:
    """Derive relative weights for each indicator from the data.

    Currently supports only the ``"correlation"`` method, which
    computes the absolute correlation between each indicator and the
    sum of all indicators.  If no variation is present, equal weights
    are returned.
    """
    if method != "correlation":
        raise ValueError(f"Unsupported weight derivation method: {method}")
    sub = df[list(indicators)].dropna()
    if sub.empty:
        n = len(list(indicators))
        return {col: 1.0 / n for col in indicators}
    pseudo = sub.sum(axis=1)
    corrs = []
    for col in indicators:
        x = sub[col].to_numpy(dtype=float)
        y = pseudo.to_numpy(dtype=float)
        if np.std(x) == 0 or np.std(y) == 0:
            corrs.append(0.0)
        else:
            corr = float(np.corrcoef(x, y)[0, 1])
            corrs.append(abs(corr))
    weights = np.array(corrs, dtype=float)
    if weights.sum() == 0:
        n = len(corrs)
        return {col: 1.0 / n for col in indicators}
    weights /= weights.sum()
    return {col: float(w) for col, w in zip(indicators, weights)}

def compute_risk_index(
    df: pd.DataFrame,
    indicators: Iterable[str] = INDICATOR_COLUMNS,
    weights: Optional[Dict[str, float]] = None,
    method: str = "equal",
) -> pd.Series:
    """Compute a composite drought risk index as a weighted combination.

    The index is scaled to the range 0–100 for interpretability.  If
    ``weights`` is not provided, ``method`` determines how weights are
    derived: ``"equal"`` assigns equal weights to all indicators,
    ``"correlation"`` derives weights based on correlations with the
    sum of indicators.
    """
    df_ind = df[list(indicators)].astype(float)
    if weights is None:
        if method == "equal":
            n = len(list(indicators))
            weights_arr = np.ones(n) / n
            weights = {col: float(w) for col, w in zip(indicators, weights_arr)}
        elif method == "correlation":
            weights = derive_weights(df, indicators=indicators, method="correlation")
        else:
            raise ValueError(f"Unknown weighting method: {method}")
    else:
        total = float(sum(weights.values())) or 1.0
        weights = {k: float(v) / total for k, v in weights.items()}
    values = np.zeros(len(df), dtype=float)
    mask = np.zeros(len(df), dtype=bool)
    for col in indicators:
        w = weights.get(col, 0.0)
        if w == 0:
            continue
        col_vals = df[col].astype(float)
        is_na = col_vals.isna().to_numpy()
        values[~is_na] += col_vals[~is_na].to_numpy() * w
        mask |= is_na
    valid_values = values[~mask]
    if valid_values.size == 0 or np.max(valid_values) - np.min(valid_values) == 0:
        return pd.Series(0.0, index=df.index)
    min_val = float(np.nanmin(valid_values))
    max_val = float(np.nanmax(valid_values))
    scaled = 100.0 * (values - min_val) / (max_val - min_val)
    return pd.Series(scaled, index=df.index)

def compute() -> "xr.DataArray":
    """Legacy helper to compute the risk index and write it to NetCDF.

    This function retains backwards compatibility with older workflows
    that operated on ``indicators.nc`` and wrote ``risk_index.nc``.
    It uses xarray to compute an equally weighted risk index.
    """
    import xarray as xr  # type: ignore
    from src.config import DATA_DIR

    ind = xr.open_dataset(DATA_DIR / "indicators.nc")
    risk = (ind.rain_deficit + ind.ndvi_stress + ind.soil_dryness + ind.temp_anomaly) / 4.0
    risk = risk.rename("risk_index")
    risk.to_netcdf(DATA_DIR / "risk_index.nc")
    return risk

__all__ = [
    "INDICATOR_COLUMNS",
    "derive_weights",
    "compute_risk_index",
    "compute",
]
