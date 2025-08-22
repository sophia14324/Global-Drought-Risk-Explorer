from __future__ import annotations

import json
import urllib.parse
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import altair as alt
import pydeck as pdk


ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
ASSETS_DIR = ROOT / "assets"
DATA_DIR.mkdir(parents=True, exist_ok=True)
ASSETS_DIR.mkdir(parents=True, exist_ok=True)

PARQUET = DATA_DIR / "admin_monthly_indicators.parquet"
CSV_DIR = DATA_DIR / "gee_monthly"
NDVI_CSV = CSV_DIR / "EA_admin1_monthly_NDVI.csv"
RAIN_CSV = CSV_DIR / "EA_admin1_monthly_CHIRPS.csv"
SMAP_CSV = CSV_DIR / "EA_admin1_monthly_SMAP_RZSM.csv"

WORLD_GJ = ASSETS_DIR / "world_admin0.geojson"           
EA_ADMIN1_GJ = ASSETS_DIR / "east_africa_admin1.geojson" 
ISO_CSV = ASSETS_DIR / "country_iso.csv"                  


st.set_page_config(
    page_title="Global Drought Risk Explorer",
    page_icon="🌍",
    layout="wide",
)

st.markdown(
    """
    <style>
    .kpi-card {
        padding: 14px 16px;
        border-radius: 14px;
        background: #ffffff;
        box-shadow: 0 1px 2px rgba(0,0,0,0.06), 0 8px 24px rgba(0,0,0,0.06);
        border: 1px solid #eee;
    }
    .kpi-title { font-size: 0.9rem; color: #555; margin-bottom: 6px; }
    .kpi-value { font-size: 1.6rem; font-weight: 700; color: #111; }
    .kpi-sub   { font-size: 0.85rem; color: #666; }
    .soft-box {
        padding: 12px 14px; border-radius: 12px; background:#fff;
        box-shadow: 0 1px 2px rgba(0,0,0,.05);
        border: 1px solid #eee;
    }
    .tabbar { position: sticky; top: 0; background: var(--background-color); z-index: 5; padding-top: 6px; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner=False)
def _load_core() -> pd.DataFrame:
    """Load indicators from Parquet or build from CSVs (tiny)."""
    if PARQUET.exists():
        df = pd.read_parquet(PARQUET)
    elif NDVI_CSV.exists() and RAIN_CSV.exists() and SMAP_CSV.exists():
        def _read(path, val):
            d = pd.read_csv(path)
            d["date"] = pd.to_datetime(d["date"])
            return d[["ADM0_NAME","ADM1_NAME","date","value"]].rename(columns={"value": val})
        ndvi = _read(NDVI_CSV, "ndvi")
        rain = _read(RAIN_CSV, "rain_mm")
        rzsm = _read(SMAP_CSV, "rzsm")

        def zcol(d, col):
            d = d.copy()
            d["month"] = d["date"].dt.month
            def _z(s):
                sd = s.std(ddof=0)
                if not np.isfinite(sd) or sd == 0: return pd.Series(0.0, index=s.index)
                return (s - s.mean())/sd
            d["z"] = d.groupby(["ADM0_NAME","ADM1_NAME","month"])[col].transform(_z)
            return d

        ndvi = zcol(ndvi, "ndvi"); ndvi["ndvi_stress"] = -ndvi["z"]
        rain = zcol(rain, "rain_mm"); rain["rain_deficit"] = -rain["z"]
        rzsm = zcol(rzsm, "rzsm");    rzsm["soil_dryness"] = -rzsm["z"]

        keys = ["ADM0_NAME","ADM1_NAME","date"]
        df = (ndvi[keys+["ndvi_stress"]]
              .merge(rain[keys+["rain_deficit"]], on=keys, how="outer")
              .merge(rzsm[keys+["soil_dryness"]], on=keys, how="outer")
              .sort_values(keys).reset_index(drop=True))
        df["risk_index"] = df[["ndvi_stress","rain_deficit","soil_dryness"]].mean(axis=1, skipna=True)
        df["risk_index_sm3"] = (df.groupby(["ADM0_NAME","ADM1_NAME"])["risk_index"]
                                  .transform(lambda s: s.rolling(3, min_periods=1).mean()))
        try:
            df.to_parquet(PARQUET, index=False)
        except Exception:
            pass
    else:
        st.error(
            "Data not found.\n\nAdd `data/admin_monthly_indicators.parquet` **or** the three CSVs in `data/gee_monthly/`:\n"
            "- EA_admin1_monthly_NDVI.csv\n- EA_admin1_monthly_CHIRPS.csv\n- EA_admin1_monthly_SMAP_RZSM.csv"
        )
        st.stop()

    df = df.rename(columns={"ADM0_NAME":"Country","ADM1_NAME":"Area"})
    df = df.sort_values(["Country","Area","date"]).reset_index(drop=True)
    return df

@st.cache_data(show_spinner=False)
def _load_geometries():
    """
    Try global admin0 first (ISO-3 key); else use EA admin1 (ADM0/ADM1 names).
    Returns (geojson, level, id_field, iso_key)
    level: "world" or "ea_admin1"
    """
    if WORLD_GJ.exists():
        gj = json.loads(WORLD_GJ.read_text(encoding="utf-8"))
        # Heuristic: find an ISO3-ish property
        candidates = ["ISO_A3","ADM0_A3","iso_a3","ADM0_A3_US","WB_A3","ISO3"]
        sample_props = gj.get("features",[{}])[0].get("properties",{})
        iso_key = next((k for k in candidates if k in sample_props), None)
        return gj, "world", None, iso_key
    elif EA_ADMIN1_GJ.exists():
        gj = json.loads(EA_ADMIN1_GJ.read_text(encoding="utf-8"))
        return gj, "ea_admin1", None, None
    else:
        return None, None, None, None

@st.cache_data(show_spinner=False)
def _load_iso_map():
    """Optional country ISO map with Continent; columns: ADM0_NAME, ISO3, Continent."""
    if ISO_CSV.exists():
        m = pd.read_csv(ISO_CSV).dropna(subset=["ADM0_NAME","ISO3"])
        m["ADM0_NAME"] = m["ADM0_NAME"].astype(str)
        m["ISO3"] = m["ISO3"].astype(str)
        return m
    return pd.DataFrame(columns=["ADM0_NAME","ISO3","Continent"])

def to_0_100(df: pd.DataFrame, col="risk_index_sm3") -> pd.Series:
    """Relative 0–100 index by date using percent ranks (higher = worse)."""
    pct = df.groupby("date")[col].rank(pct=True, na_option="keep")
    return (pct * 100).round(1)

def category(val: float) -> str:
    if pd.isna(val): return "No data"
    if val < 20:  return "Low"
    if val < 40:  return "Medium"
    if val < 70:  return "High"
    return "Extreme"

def blue_green_yellow_orange_red_classes():
  
    return [
        [33, 102, 172, 200],   # deep blue
        [67, 147, 195, 200],   # blue
        [146, 197, 222, 200],  # blue-teal
        [199, 233, 180, 200],  # greenish
        [255, 255, 153, 200],  # yellow
        [253, 174, 97, 200],   # orange
        [215, 48, 39, 200],    # red
    ]

def quantile_bins(values: pd.Series, k=7):
    vals = values.dropna().to_numpy()
    if len(vals) == 0:
        return []
    qs = np.linspace(0, 1, k+1)
    edges = np.quantile(vals, qs)
    edges = np.unique(edges)
    return edges


with st.spinner("Loading data…"):
    df = _load_core()
    # Build 0–100 scale
    df["DRI_0_100"] = to_0_100(df, "risk_index_sm3")
    last_date = pd.to_datetime(df["date"].max())

    iso_map = _load_iso_map()
    if not iso_map.empty:
        df = df.merge(iso_map.rename(columns={"ADM0_NAME":"Country"}), on="Country", how="left")
    else:
        df["ISO3"] = None
        df["Continent"] = None

    latest = df[df["date"] == last_date].copy()
    prev_date = df["date"].sort_values().unique()[-2] if df["date"].nunique() > 1 else last_date
    prev = df[df["date"] == prev_date][["Country","Area","DRI_0_100"]].rename(columns={"DRI_0_100":"DRI_prev"})
    latest = latest.merge(prev, on=["Country","Area"], how="left")
    latest["MoM"] = latest["DRI_0_100"] - latest["DRI_prev"]
    # YoY
    yoy_date = last_date - pd.DateOffset(years=1)
    yoy = df[df["date"] == yoy_date][["Country","Area","DRI_0_100"]].rename(columns={"DRI_0_100":"DRI_yoy_base"})
    latest = latest.merge(yoy, on=["Country","Area"], how="left")
    latest["YoY"] = latest["DRI_0_100"] - latest["DRI_yoy_base"]
    latest["Category"] = latest["DRI_0_100"].apply(category)


col1, col2 = st.columns([0.72, 0.28])
with col1:
    st.markdown("### **Global Drought Risk Explorer**")
    st.markdown(
        "The **Drought Risk Index** blends recent climate signals and exposure on a **0–100** scale. "
        "**Higher = higher risk.** Updated monthly.",
        help="The Drought Risk Index combines recent climate signals (vegetation stress, rainfall deficit, soil dryness) and exposure. "
             "Values are ranked by month across areas to a 0–100 scale (percentile)."
    )
with col2:
    st.markdown(f"**Last updated:** {last_date.strftime('%b %Y')}")
    st.markdown(
        "[![CHIRPS](https://img.shields.io/badge/CHIRPS-precip-0b7285?style=flat-square)](#) "
        "[![MODIS](https://img.shields.io/badge/MODIS-NDVI-2d6a4f?style=flat-square)](#) "
        "[![SMAP](https://img.shields.io/badge/SMAP-soil-6a4c93?style=flat-square)](#)"
    )

qp = st.query_params 
def set_query_params(country=None, metric=None, date=None, search=None):
    params = {}
    if country: params["country"] = country
    if metric:  params["metric"]  = metric
    if date:    params["date"]    = date
    if search:  params["q"]       = search
    st.query_params.clear()
    st.query_params.update(params)


st.sidebar.header("Filters")
countries = sorted(latest["Country"].dropna().unique().tolist())
country_sel = st.sidebar.multiselect("Countries", countries, default=countries)
search_q = st.sidebar.text_input("Search area", qp.get("q", ""))
metric_sel = st.sidebar.selectbox(
    "Metric",
    ["DRI_0_100","ndvi_stress","rain_deficit","soil_dryness","risk_index_sm3"],
    index=0,
    help="DRI_0_100 is the 0–100 scaled index (higher=worse). Others are standardized components."
)

set_query_params(
    country=",".join(country_sel) if country_sel else None,
    metric=metric_sel,
    date=last_date.strftime("%Y-%m"),
    search=search_q or None
)

mask = latest["Country"].isin(country_sel) if country_sel else slice(None)
snap = latest.loc[mask].copy()
if search_q:
    s = search_q.lower()
    snap = snap[snap["Area"].str.lower().str.contains(s) | snap["Country"].str.lower().str.contains(s)]

dfc = df[df["Country"].isin(country_sel)] if country_sel else df


def kpi(col, title, value, sub=None):
    with col:
        st.markdown(f'<div class="kpi-card"><div class="kpi-title">{title}</div>'
                    f'<div class="kpi-value">{value}</div>'
                    f'<div class="kpi-sub">{sub or ""}</div></div>', unsafe_allow_html=True)

k1, k2, k3, k4 = st.columns(4)
avg_risk = f"{snap['DRI_0_100'].mean():.1f}" if not snap.empty else "—"
pct_high = (snap["DRI_0_100"]>=70).mean()*100 if not snap.empty else np.nan
countries_count = dfc["Country"].nunique()
mom = (snap["MoM"].mean() if "MoM" in snap else np.nan)
kpi(k1, "Avg Risk (selection)", avg_risk)
kpi(k2, "% Area High Risk (≥70)", f"{pct_high:.1f}%" if np.isfinite(pct_high) else "—")
kpi(k3, "Countries Monitored", f"{countries_count}")
kpi(k4, "30-day Change", f"{mom:+.1f}" if np.isfinite(mom) else "—")


st.markdown('<div class="tabbar"></div>', unsafe_allow_html=True)
tab_map, tab_trend, tab_table = st.tabs(["🗺️ Map", "📈 Trends", "📋 Table"])


with tab_map:
    st.markdown("#### Choropleth")
    # Try geometries
    gj, level, _, iso_key = _load_geometries()
    if gj is None:
        st.info("No geometry found. Add `assets/world_admin0.geojson` (with ISO-3) or `assets/east_africa_admin1.geojson`.")
    else:
        if level == "world":
            if dfc["ISO3"].notna().any():
                agg = (dfc[dfc["date"]==last_date]
                        .groupby(["Country","ISO3"])[metric_sel]
                        .mean()
                        .reset_index())
            else:
                agg = (dfc[dfc["date"]==last_date]
                        .groupby(["Country"])[metric_sel]
                        .mean()
                        .reset_index())
                agg["ISO3"] = None
            val_map = {r["ISO3"]:(r["Country"], float(r[metric_sel])) for _,r in agg.iterrows()}
            bins = quantile_bins(agg[metric_sel], k=7)
            palette = blue_green_yellow_orange_red_classes()

            feats = gj.get("features", [])
            for f in feats:
                props = f.get("properties", {})
                key_iso = props.get(iso_key) if iso_key else None
                country, val = (None, None)
                if key_iso in val_map:
                    country, val = val_map[key_iso]
                props["Country"] = country or props.get("NAME","")
                props["DRI_value"] = val

                if val is None or (len(bins)==0):
                    props["fill_rgba"] = [200,200,200,120]
                    props["class_label"] = "No data"
                else:
                    idx = int(np.digitize([val], bins, right=False)[0] - 1)
                    idx = max(0, min(idx, len(palette)-1))
                    props["fill_rgba"] = palette[idx]
                    props["class_label"] = f"{bins[idx]:.1f}–{bins[min(idx+1,len(bins)-1)]:.1f}"
        else:
            snap_map = { (r["Country"], r["Area"]): float(r[metric_sel]) if pd.notna(r[metric_sel]) else None
                         for _, r in snap.iterrows() }
            feats = gj.get("features", [])
            vals = []
            for f in feats:
                p = f.get("properties", {})
                key = (p.get("ADM0_NAME"), p.get("ADM1_NAME"))
                val = snap_map.get(key)
                p["Country"] = p.get("ADM0_NAME")
                p["Area"] = p.get("ADM1_NAME")
                p["DRI_value"] = val
                vals.append(val)
            vals_series = pd.Series([v for v in vals if v is not None])
            bins = quantile_bins(vals_series, k=7)
            palette = blue_green_yellow_orange_red_classes()
            for f in feats:
                p = f.get("properties", {})
                val = p.get("DRI_value")
                if val is None or (len(bins)==0):
                    p["fill_rgba"] = [200,200,200,120]
                    p["class_label"] = "No data"
                else:
                    idx = int(np.digitize([val], bins, right=False)[0] - 1)
                    idx = max(0, min(idx, len(palette)-1))
                    p["fill_rgba"] = palette[idx]
                    p["class_label"] = f"{bins[idx]:.1f}–{bins[min(idx+1,len(bins)-1)]:.1f}"

        layer = pdk.Layer(
            "GeoJsonLayer",
            gj,
            pickable=True,
            stroked=False,
            filled=True,
            get_fill_color="properties.fill_rgba",
            get_line_color=[90,90,90,120],
            line_width_min_pixels=0.5,
        )
        view_state = pdk.ViewState(latitude=1.5, longitude=30.0, zoom=2.6 if level=="world" else 4)
        st.pydeck_chart(pdk.Deck(layers=[layer], initial_view_state=view_state, map_style=None))
        st.caption("Legend: 0, 20, 40, 60, 80, 100 — higher = worse. Quantile-binned colors; gray = no data.")



with tab_trend:
    st.markdown("#### Portfolio Risk Over Time")
    toggle = st.segmented_control("Series", options=["Mean","P90","Assets-weighted"], default="Mean")
    ma3 = st.checkbox("3-month moving average", value=True)
    grp = dfc.groupby("date")["DRI_0_100"]
    if toggle == "Mean":
        ts = grp.mean().reset_index(name="DRI")
    elif toggle == "P90":
        ts = grp.quantile(0.9).reset_index(name="DRI")
    else:
        if "asset_weight" in dfc.columns:
            w = (dfc.dropna(subset=["asset_weight","DRI_0_100"])
                   .groupby("date")
                   .apply(lambda g: np.average(g["DRI_0_100"], weights=g["asset_weight"]))
                   .reset_index(name="DRI"))
            ts = w
        else:
            ts = grp.mean().reset_index(name="DRI")
            st.caption("No asset weights found; using mean.")

    if ma3:
        ts["DRI_MA3"] = ts["DRI"].rolling(3, min_periods=1).mean()
        yfield = "DRI_MA3"
    else:
        yfield = "DRI"

    base = alt.Chart(ts).mark_line().encode(
        x=alt.X("date:T", title="Month"),
        y=alt.Y(f"{yfield}:Q", title="Drought Risk Index (0–100)"),
        tooltip=[alt.Tooltip("date:T"), alt.Tooltip(f"{yfield}:Q", format=".1f")]
    ).properties(height=280)
    st.altair_chart(base, use_container_width=True)
    st.caption("Tip: This is the average (or hotspot P90) risk over your current selection.")

with tab_table:
    st.markdown("#### Areas — Latest View")
    table = snap[["Country","Area","DRI_0_100","MoM","YoY","Category"]].copy()
    table = table.rename(columns={
        "Country": "Country / Region",
        "Area": "Area",
        "DRI_0_100":"Latest Index",
        "MoM":"MoM",
        "YoY":"YoY",
        "Category":"Category"
    }).sort_values("Latest Index", ascending=False)
    st.dataframe(table, use_container_width=True, hide_index=True)
    # Download
    csv_bytes = table.to_csv(index=False).encode("utf-8")
    st.download_button("Download CSV", data=csv_bytes, file_name="drought_risk_latest.csv", mime="text/csv")


st.divider()
c1, c2 = st.columns([0.6, 0.4])
with c1:
    params = st.query_params
    base_url = st.request.url if hasattr(st, "request") else ""
    share_url = base_url.split("?")[0] + "?" + urllib.parse.urlencode(params, doseq=True) if base_url else ""
    st.markdown("##### Share")
    st.text_input("Copy link with current filters", share_url, label_visibility="collapsed")
with c2:
    st.markdown("##### About the Index")
    st.info(
        "The **Drought Risk Index** combines recent climate signals and exposure on a **0–100** scale. "
        "Higher numbers mean higher risk. Values are updated monthly and normalized within each month across areas. "
        "Sources: CHIRPS (rainfall), MODIS NDVI (vegetation), SMAP (root-zone soil moisture).",
        icon="ℹ️",
    )
