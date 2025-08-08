import streamlit as st, xarray as xr, geopandas as gpd, pandas as pd
import leafmap.foliumap as leafmap
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.config import DATA_DIR, REGION_SHAPE

st.set_page_config(page_title="EA Climate-Risk", layout="wide")
st.title("East Africa: Multi-Hazard Climate Risk Dashboard")

risk = xr.open_dataset(DATA_DIR / "risk_index.nc").risk_index
clusters = xr.open_dataset(DATA_DIR / "clusters.nc").cluster_label
admin = gpd.read_file(REGION_SHAPE).to_crs("EPSG:4326")

month = st.slider("Select month", 0, len(risk.time)-1, value=len(risk.time)-1)
hazard = st.selectbox("Hazard component", ["risk_index",
                                           "rain_deficit",
                                           "ndvi_stress",
                                           "soil_dryness",
                                           "temp_anomaly"])

with st.container():
    m = leafmap.Map(center=[0, 35], zoom=4)
    m.add_basemap("HYBRID")
    da = risk.isel(time=month)
    m.add_heatmap(da.lon, da.lat, da.values)
    m.add_gdf(admin, style={"fillOpacity": 0, "color": "#333", "weight": 1})
    st.components.v1.html(m.to_html(), height=600)
    
st.line_chart(risk.mean(dim=["lat", "lon"]).to_series(), height=200)

if risk.isel(time=month).mean() > 1.5:
    st.error("⚠️  Critical risk threshold breached!")
