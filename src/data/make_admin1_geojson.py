# src/data/make_admin1_geojson.py
# Build East Africa admin-1 polygons as a local GeoJSON for the dashboard.

from pathlib import Path
import ee

from src.config import ASSETS_DIR

# Initialize EE
EE_PROJECT = "operating-axis-468409-s4"
ee.Initialize(project=EE_PROJECT)

# Countries to include (GAUL names)
EA = [
    "Kenya","Uganda","Tanzania","Ethiopia","Somalia",
    "South Sudan","Rwanda","Burundi","Sudan","Eritrea","Djibouti"
]

# GAUL administrative boundaries
gaul1 = ee.FeatureCollection("FAO/GAUL/2015/level1").filter(
    ee.Filter.inList("ADM0_NAME", EA)
)

# Optional: simplify geometry to keep the GeoJSON light-weight.
# NOTE: EE Geometry.simplify only accepts maxError (meters).
def _simplify(f):
    return f.setGeometry(f.geometry().simplify(1000))  # ~1 km tolerance

gaul1_simplified = gaul1.map(_simplify)

# Ensure output folder exists
ASSETS_DIR.mkdir(parents=True, exist_ok=True)
out = ASSETS_DIR / "east_africa_admin1.geojson"

# Try direct local export with geemap; fall back to Drive if needed
try:
    import geemap
    geemap.ee_export_vector(gaul1_simplified, filename=str(out))
    print("✅ Wrote", out)
except Exception as e:
    print("Local export failed, exporting to Drive instead… Reason:", e)
    task = ee.batch.Export.table.toDrive(
        collection=gaul1_simplified,
        description="EA_admin1_geojson",
        folder="gee_monthly",
        fileFormat="GeoJSON",
    )
    task.start()
    print("▶️ Drive export started. When it completes, download the file and save it as:")
    print("   ", out)
