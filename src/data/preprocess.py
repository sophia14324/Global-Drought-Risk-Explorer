"""Data preprocessing for the Global Drought Risk Explorer.

This module assembles monthly indicators exported from Google Earth
Engine into a tidy table, standardises each indicator within calendar
months and computes composite drought risk indices.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

# Import configuration and risk index utilities from the src package.
from src.config import DATA_DIR, START, END, normalised_weights  # type: ignore
from src.modelling.risk_index import compute_risk_index, INDICATOR_COLUMNS  # type: ignore

# Paths to the exported CSVs
GEE_DIR: Path = DATA_DIR / "gee_monthly"
NDVI_CSV: Path = GEE_DIR / "EA_admin1_monthly_NDVI.csv"
CHIRPS_CSV: Path = GEE_DIR / "EA_admin1_monthly_CHIRPS.csv"
SMAP_CSV: Path = GEE_DIR / "EA_admin1_monthly_SMAP_RZSM.csv"

def _require_files(paths: List[Path]) -> None:
    """Ensure that required CSV files exist."""
    missing = [str(p) for p in paths if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing required CSV(s):\n " + "\n ".join(missing) +
            "\nCopy the completed Drive exports into data/gee_monthly/."
        )

def _read_csv(path: Path, value_name: str) -> pd.DataFrame:
    """Read a GEE CSV and return a tidy DataFrame."""
    df = pd.read_csv(path)
    for col in ("ADM0_NAME", "ADM1_NAME", "date", "value"):
        if col not in df.columns:
            raise ValueError(f"{path.name} missing required column: {col}")
    df["date"] = pd.to_datetime(df["date"])
    mask = (df["date"] >= pd.to_datetime(START)) & (df["date"] <= pd.to_datetime(END))
    df = df.loc[mask, ["ADM0_NAME", "ADM1_NAME", "date", "value"]].rename(
        columns={"value": value_name}
    )
    return df

def _monthly_zscore(df: pd.DataFrame, value_col: str) -> pd.DataFrame:
    """Compute a seasonal z-score within each (country, area, calendar month)."""
    x = df.copy()
    x["month"] = x["date"].dt.month
    def _z_transform(s: pd.Series) -> pd.Series:
        m = s.mean()
        sd = s.std(ddof=0)
        if not np.isfinite(sd) or sd == 0:
            return pd.Series(0.0, index=s.index)
        return (s - m) / sd
    x["z"] = (
        x.groupby(["ADM0_NAME", "ADM1_NAME", "month"])[value_col]
        .transform(_z_transform)
    )
    return x

def build_indicators(weights: Optional[Dict[str, float]] = None) -> pd.DataFrame:
    """Build the indicators table and compute risk indices."""
    _require_files([NDVI_CSV, CHIRPS_CSV, SMAP_CSV])
    ndvi = _read_csv(NDVI_CSV, "ndvi")
    rain = _read_csv(CHIRPS_CSV, "rain_mm")
    rzsm = _read_csv(SMAP_CSV, "rzsm")

    ndvi_z = _monthly_zscore(ndvi, "ndvi")
    rain_z = _monthly_zscore(rain, "rain_mm")
    rzsm_z = _monthly_zscore(rzsm, "rzsm")

    ndvi_z["ndvi_stress"] = -ndvi_z["z"]
    rain_z["rain_deficit"] = -rain_z["z"]
    rzsm_z["soil_dryness"] = -rzsm_z["z"]

    keys = ["ADM0_NAME", "ADM1_NAME", "date"]
    df = (
        ndvi_z[keys + ["ndvi_stress"]]
        .merge(rain_z[keys + ["rain_deficit"]], on=keys, how="outer")
        .merge(rzsm_z[keys + ["soil_dryness"]], on=keys, how="outer")
        .sort_values(keys)
        .reset_index(drop=True)
    )

    # Normalise weights and prepare mapping
    w = normalised_weights(weights)
    df_ind = df.copy()
    df_ind["rain_deficit_z"] = df_ind["rain_deficit"]
    df_ind["ndvi_stress_z"] = df_ind["ndvi_stress"]
    df_ind["soil_moisture_z"] = df_ind["soil_dryness"]
    df_ind["temp_anomaly_z"] = np.nan

    modelling_weights = {
        "rain_deficit_z": w.get("rain_deficit", 0.0),
        "ndvi_stress_z": w.get("ndvi_stress", 0.0),
        "soil_moisture_z": w.get("soil_dryness", 0.0),
        "temp_anomaly_z": w.get("temp_anomaly", 0.0),
    }

    df["risk_index"] = compute_risk_index(
        df_ind,
        indicators=INDICATOR_COLUMNS,
        weights=modelling_weights,
        method="equal",
    )
    df["risk_index_sm3"] = (
        df.groupby(["ADM0_NAME", "ADM1_NAME"])["risk_index"]
        .transform(lambda s: s.rolling(3, min_periods=1).mean())
    )

    out_parquet = DATA_DIR / "admin_monthly_indicators.parquet"
    out_csv = DATA_DIR / "admin_monthly_indicators.csv"
    df.to_parquet(out_parquet, index=False)
    df.to_csv(out_csv, index=False)
    print(f"✅ Wrote:\n  {out_parquet}\n  {out_csv}")
    return df

if __name__ == "__main__":
    build_indicators()
