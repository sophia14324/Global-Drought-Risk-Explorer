# Standalone Streamlit app (no imports from src.*)
import sys, json
from pathlib import Path

import pandas as pd
import streamlit as st
import pydeck as pdk

# --- Paths (repo layout: <root>/{data,assets,src}) ---
ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
ASSETS_DIR = ROOT / "assets"
DATA_DIR.mkdir(parents=True, exist_ok=True)
(ASSETS_DIR).mkdir(parents=True, exist_ok=True)

st.set_page_config(page_title="EA Climate Risk (Admin-1)", layout="wide")
st.title("East Africa — Admin-level Climate Risk Dashboard")

PARQUET = DATA_DIR / "admin_monthly_indicators.parquet"
CSV_DIR = DATA_DIR / "gee_monthly"
NDVI_CSV   = CSV_DIR / "EA_admin1_monthly_NDVI.csv"
CHIRPS_CSV = CSV_DIR / "EA_admin1_monthly_CHIRPS.csv"
SMAP_CSV   = CSV_DIR / "EA_admin1_monthly_SMAP_RZSM.csv"

def _monthly_z(x: pd.Series) -> pd.Series:
    m, sd = x.mean(), x.std(ddof=0)
    if not pd.notna(sd) or sd == 0:
        return pd.Series(0.0, index=x.index)
    return (x - m) / sd

def _read_csv(path: Path, valname: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    for c in ("ADM0_NAME","ADM1_NAME","date","value"):
        if c not in df.columns:
            raise ValueError(f"{path.name} missing column: {c}")
    df["date"] = pd.to_datetime(df["date"])
    return df[["ADM0_NAME","ADM1_NAME","date","value"]].rename(columns={"value": valname})

def _build_from_csvs() -> pd.DataFrame:
    ndvi = _read_csv(NDVI_CSV, "ndvi")
    rain = _read_csv(CHIRPS_CSV, "rain_mm")
    rzsm = _read_csv(SMAP_CSV, "rzsm")

    for d, col in ((ndvi,"ndvi"), (rain,"rain_mm"), (rzsm,"rzsm")):
        d["month"] = d["date"].dt.month
        d["z"] = d.groupby(["ADM0_NAME","ADM1_NAME","month"])[col].transform(_monthly_z)

    ndvi["ndvi_stress"]  = -ndvi["z"]
    rain["rain_deficit"] = -rain["z"]
    rzsm["soil_dryness"] = -rzsm["z"]

    keys = ["ADM0_NAME","ADM1_NAME","date"]
    df = (ndvi[keys+["ndvi_stress"]]
          .merge(rain[keys+["rain_deficit"]], on=keys, how="outer")
          .merge(rzsm[keys+["soil_dryness"]], on=keys, how="outer")
          .sort_values(keys).reset_index(drop=True))

    df["risk_index"] = df[["ndvi_stress","rain_deficit","soil_dryness"]].mean(axis=1, skipna=True)
    df["risk_index_sm3"] = (df.groupby(["ADM0_NAME","ADM1_NAME"])["risk_index"]
                              .transform(lambda s: s.rolling(3, min_periods=1).mean()))
    return df

# ---- Load data (Parquet -> CSVs -> ask for upload) ----
if PARQUET.exists():
    df = pd.read_parquet(PARQUET)
elif NDVI_CSV.exists() and CHIRPS_CSV.exists() and SMAP_CSV.exists():
    df = _build_from_csvs()
    df.to_parquet(PARQUET, index=False)
else:
    st.error(
        "Data not found.\n\nAdd `data/admin_monthly_indicators.parquet` **or** the three CSVs in `data/gee_monthly/`:\n"
        "- EA_admin1_monthly_NDVI.csv\n- EA_admin1_monthly_CHIRPS.csv\n- EA_admin1_monthly_SMAP_RZSM.csv"
    )
    st.stop()

df = df.sort_values(["ADM0_NAME","ADM1_NAME","date"]).reset_index(drop=True)

# --- Sidebar filters ---
st.sidebar.header("Filters")
countries = sorted(df["ADM0_NAME"].dropna().unique().tolist())
country_sel = st.sidebar.multiselect("Countries", countries, default=countries)
metric = st.sidebar.selectbox(
    "Metric",
    ["risk_index_sm3","risk_index","ndvi_stress","rain_deficit","soil_dryness"],
    index=0,
)

dfc = df[df["ADM0_NAME"].isin(country_sel)] if country_sel else df
t_min, t_max = dfc["date"].min(), dfc["date"].max()
date_sel = st.slider("Month", min_value=t_min.to_pydatetime(), max_value=t_max.to_pydatetime(),
                     value=t_max.to_pydatetime(), format="YYYY-MM")
dsel = pd.to_datetime(date_sel)

# --- Snapshot tables ---
snap = dfc[dfc["date"] == dsel][["ADM0_NAME","ADM1_NAME",metric]].sort_values(metric, ascending=False)
c1, c2 = st.columns([1,1], gap="large")
with c1:
    st.subheader(f"Highest {metric} — {dsel.strftime('%Y-%m')}")
    st.dataframe(snap.head(25), use_container_width=True)
with c2:
    st.subheader(f"Lowest {metric} — {dsel.strftime('%Y-%m')}")
    st.dataframe(snap.tail(25), use_container_width=True)

# --- Portfolio trend ---
st.subheader("Portfolio trend")
ts = (dfc.groupby("date")[metric].mean().reset_index().sort_values("date"))
st.line_chart(ts, x="date", y=metric, height=220)

# --- Choropleth (optional) ---
admin_geojson = ASSETS_DIR / "east_africa_admin1.geojson"
if admin_geojson.exists():
    gj = json.loads(admin_geojson.read_text(encoding="utf-8"))

    snap_map = {
        (row.ADM0_NAME, row.ADM1_NAME): getattr(row, metric)
        for row in snap.itertuples(index=False)
    }
    for feat in gj.get("features", []):
        props = feat.get("properties", {})
        key = (props.get("ADM0_NAME"), props.get("ADM1_NAME"))
        val = snap_map.get(key)
        props[metric] = float(val) if val is not None and pd.notna(val) else None

    def color_expr(mname):
        return [
            "case",
            ["==", ["get", mname], None], [200,200,200,120],
            ["<", ["get", mname], -1.0], [33,102,172,180],
            ["<", ["get", mname], -0.5], [67,147,195,180],
            ["<", ["get", mname], 0.0],  [146,197,222,180],
            ["<", ["get", mname], 0.5],  [244,165,130,180],
            ["<", ["get", mname], 1.0],  [214,96,77,180],
            [215,48,39,200],
        ]

    layer = pdk.Layer(
        "GeoJsonLayer",
        gj,
        pickable=True,
        stroked=False,
        filled=True,
        get_fill_color=color_expr(metric),
        get_line_color=[80,80,80,120],
        line_width_min_pixels=0.5,
    )
    view_state = pdk.ViewState(latitude=1.0, longitude=33.0, zoom=3.8)
    st.subheader("Choropleth")
    st.pydeck_chart(pdk.Deck(layers=[layer], initial_view_state=view_state, map_style=None))
else:
    st.info("No admin polygons found. Add `assets/east_africa_admin1.geojson` to enable the map.")
