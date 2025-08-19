# src/data/preprocess.py
"""
Build admin-level monthly indicators (CSV -> tidy table) with a composite risk index.
Inputs: 3 small CSVs exported from Earth Engine to data/gee_monthly/
Outputs: data/admin_monthly_indicators.parquet  (and CSV)

Indicators:
- ndvi_stress     = - Z(NDVI)              (lower NDVI => higher stress)
- rain_deficit    = - Z(CHIRPS_mm)         (lower rain => higher deficit)
- soil_dryness    = - Z(SMAP_RZSM)         (lower soil moisture => drier)

Composite:
- risk_index      = mean([ndvi_stress, rain_deficit, soil_dryness])
- risk_index_sm3  = 3-month rolling mean per admin-1
"""

from pathlib import Path
import numpy as np
import pandas as pd
from src.config import DATA_DIR, START, END

GEE_DIR = DATA_DIR / "gee_monthly"
NDVI_CSV   = GEE_DIR / "EA_admin1_monthly_NDVI.csv"
CHIRPS_CSV = GEE_DIR / "EA_admin1_monthly_CHIRPS.csv"
SMAP_CSV   = GEE_DIR / "EA_admin1_monthly_SMAP_RZSM.csv"


def _require_files(paths):
    missing = [str(p) for p in paths if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing required CSV(s):\n  " + "\n  ".join(missing) +
            "\nCopy the completed Drive exports into data/gee_monthly/."
        )


def _read_csv(path: Path, value_name: str) -> pd.DataFrame:
    """Read a GEE CSV and keep essentials."""
    df = pd.read_csv(path)
    for col in ("ADM0_NAME", "ADM1_NAME", "date", "value"):
        if col not in df.columns:
            raise ValueError(f"{path.name} missing required column: {col}")
    df["date"] = pd.to_datetime(df["date"])
    mask = (df["date"] >= pd.to_datetime(START)) & (df["date"] < pd.to_datetime(END))
    df = df.loc[mask, ["ADM0_NAME", "ADM1_NAME", "date", "value"]].rename(columns={"value": value_name})
    return df


def _monthly_zscore(df: pd.DataFrame, value_col: str) -> pd.DataFrame:
    """
    Seasonal z-score within each (country, admin1, calendar month).
    Uses transform to keep row alignment; no merging needed.
    If std == 0 or NaN, z = 0 for that group.
    """
    x = df.copy()
    x["month"] = x["date"].dt.month

    def _z_transform(s: pd.Series) -> pd.Series:
        m = s.mean()
        sd = s.std(ddof=0)
        if not np.isfinite(sd) or sd == 0:
            return pd.Series(0.0, index=s.index)
        return (s - m) / sd

    x["z"] = (x.groupby(["ADM0_NAME", "ADM1_NAME", "month"])[value_col]
                .transform(_z_transform))
    return x


def build_indicators():
    _require_files([NDVI_CSV, CHIRPS_CSV, SMAP_CSV])

    # 1) Load variables
    ndvi = _read_csv(NDVI_CSV,   "ndvi")
    rain = _read_csv(CHIRPS_CSV, "rain_mm")
    rzsm = _read_csv(SMAP_CSV,   "rzsm")  # may contain nulls pre-2015-04

    # 2) Z-scores per region-month
    ndvi_z = _monthly_zscore(ndvi, "ndvi")
    rain_z = _monthly_zscore(rain, "rain_mm")
    rzsm_z = _monthly_zscore(rzsm, "rzsm")

    # 3) Direction-of-risk
    ndvi_z["ndvi_stress"]  = -ndvi_z["z"]
    rain_z["rain_deficit"] = -rain_z["z"]
    rzsm_z["soil_dryness"] = -rzsm_z["z"]

    # 4) Merge to tidy table
    keys = ["ADM0_NAME", "ADM1_NAME", "date"]
    df = (ndvi_z[keys + ["ndvi_stress"]]
          .merge(rain_z[keys + ["rain_deficit"]], on=keys, how="outer")
          .merge(rzsm_z[keys + ["soil_dryness"]], on=keys, how="outer")
          .sort_values(keys)
          .reset_index(drop=True))

    # 5) Composite + smoothing
    df["risk_index"] = df[["ndvi_stress", "rain_deficit", "soil_dryness"]].mean(axis=1, skipna=True)
    df["risk_index_sm3"] = (df
        .groupby(["ADM0_NAME","ADM1_NAME"])["risk_index"]
        .transform(lambda s: s.rolling(3, min_periods=1).mean())
    )

    # 6) Save
    out_parquet = DATA_DIR / "admin_monthly_indicators.parquet"
    out_csv     = DATA_DIR / "admin_monthly_indicators.csv"
    df.to_parquet(out_parquet, index=False)
    df.to_csv(out_csv, index=False)
    print(f"✅ Wrote:\n  {out_parquet}\n  {out_csv}")
    return df


if __name__ == "__main__":
    build_indicators()
