# src/data/gee.py
"""
Admin-level monthly climate indicators (CSV, tiny outputs):
- NDVI (MODIS/061/MOD13A3) monthly mean, scaled to [-0.2..1.0]
- Rainfall (UCSB-CHG/CHIRPS/DAILY) monthly total (mm)
- Root-zone soil moisture (NASA/SMAP/SPL4SMGP/008) monthly mean (0–1)

Exports three CSVs to Google Drive/<DRIVE_FOLDER>.
After completion, download/copy them to data/gee_monthly/ for local processing.
"""

import json
from pathlib import Path
import ee

from src.config import START, END, REGION_SHAPE, BBOX, CRS, DRIVE_FOLDER

EE_PROJECT = "operating-axis-468409-s4"
ee.Initialize(project=EE_PROJECT)

# ----------------------------
# Region & admin boundaries
# ----------------------------
def load_region_geometry():
    """Return ee.Geometry for EA. Prefer GeoJSON, else GAUL union, else BBOX."""
    p = Path(REGION_SHAPE)
    if p.exists():
        try:
            import geemap
            gj = json.loads(p.read_text(encoding="utf-8"))
            ee_obj = geemap.geojson_to_ee(gj)
            if isinstance(ee_obj, ee.geometry.Geometry):
                return ee_obj
            if isinstance(ee_obj, ee.feature.Feature):
                return ee_obj.geometry()
            if isinstance(ee_obj, ee.featurecollection.FeatureCollection):
                return ee_obj.geometry()
        except Exception:
            pass

    try:
        EA = [
            "Kenya","Uganda","Tanzania","Ethiopia","Somalia",
            "South Sudan","Rwanda","Burundi","Sudan","Eritrea","Djibouti"
        ]
        gaul0 = ee.FeatureCollection("FAO/GAUL/2015/level0")
        return gaul0.filter(ee.Filter.inList("ADM0_NAME", EA)).geometry()
    except Exception:
        pass

    return ee.Geometry.BBox(*BBOX)

REGION = load_region_geometry()

# Admin-1 polygons (for per-district stats), clipped to our region
GAUL1 = ee.FeatureCollection("FAO/GAUL/2015/level1").map(
    lambda f: f.intersection(REGION, 1)
).filterBounds(REGION)

# ----------------------------
# Helpers
# ----------------------------
def monthly_dates(start_override=None, end_override=None):
    start = ee.Date(start_override or START)
    end   = ee.Date(end_override or END)
    n = end.difference(start, "month").floor()
    return ee.List.sequence(0, n.subtract(1)).map(lambda m: start.advance(m, "month"))

def export_table(feature_collection, description, folder=DRIVE_FOLDER):
    task = ee.batch.Export.table.toDrive(
        collection=feature_collection,
        description=description,
        folder=folder,
        fileFormat="CSV",
        selectors=["ADM0_NAME", "ADM1_NAME", "date", "variable", "value"],
    )
    task.start()
    print("▶️ Started:", description, "→ Drive/", folder)

# ----------------------------
# NDVI (MODIS monthly, 1 km)  MODIS/061/MOD13A3
# Scale factor 0.0001 (DN → NDVI). :contentReference[oaicite:1]{index=1}
# ----------------------------
def fc_monthly_ndvi():
    coll = ee.ImageCollection("MODIS/061/MOD13A3").select("NDVI")
    def per_month(d):
        d = ee.Date(d)
        img = coll.filterDate(d, d.advance(1, "month")).mean().clip(REGION)
        ndvi = img.multiply(0.0001)  # scale to [-0.2..1.0]
        stats = ndvi.reduceRegions(
            collection=GAUL1, reducer=ee.Reducer.mean(),
            scale=1000, crs=CRS, tileScale=2
        )
        return stats.map(lambda f: f.set({
            "date": d.format("YYYY-MM"),
            "variable": "NDVI",
            "value": f.get("mean")
        }))
    return ee.FeatureCollection(monthly_dates().map(per_month)).flatten()

# ----------------------------
# Rainfall (CHIRPS daily → monthly mm)  UCSB-CHG/CHIRPS/DAILY :contentReference[oaicite:2]{index=2}
# ----------------------------
def fc_monthly_chirps():
    coll = ee.ImageCollection("UCSB-CHG/CHIRPS/DAILY").select("precipitation")
    def per_month(d):
        d = ee.Date(d)
        monthly_mm = coll.filterDate(d, d.advance(1, "month")).sum().clip(REGION)
        stats = monthly_mm.reduceRegions(
            collection=GAUL1, reducer=ee.Reducer.mean(),
            scale=5500, crs=CRS, tileScale=2
        )
        return stats.map(lambda f: f.set({
            "date": d.format("YYYY-MM"),
            "variable": "CHIRPS_mm",
            "value": f.get("mean")
        }))
    return ee.FeatureCollection(monthly_dates().map(per_month)).flatten()

# ----------------------------
# Root-zone Soil Moisture (SMAP L4, 3-hourly → monthly mean)
# NASA/SMAP/SPL4SMGP/008 available from 2015-03-31 onward.
# We start at 2015-04-01 (first full month). Months without images emit NaN.
# ----------------------------
def fc_monthly_smap_rzsm():
    coll = ee.ImageCollection("NASA/SMAP/SPL4SMGP/008").select("sm_rootzone")
    smap_start = "2015-04-01"  # first full month within the collection window

    def per_month(d):
        d = ee.Date(d)
        month_coll = coll.filterDate(d, d.advance(1, "month"))
        # If empty → make a single-band NaN image so reduceRegions has a band.
        img = ee.Image(ee.Algorithms.If(
            month_coll.size().gt(0),
            month_coll.mean().rename("sm_rootzone"),
            ee.Image.constant(float("nan")).rename("sm_rootzone")
        )).clip(REGION)

        stats = img.reduceRegions(
            collection=GAUL1,
            reducer=ee.Reducer.mean(),
            scale=11000,   # ~9–11 km native
            crs=CRS,
            tileScale=2
        )
        return stats.map(lambda f: f.set({
            "date": d.format("YYYY-MM"),
            "variable": "SMAP_RZSM",
            "value": f.get("mean")
        }))

    return ee.FeatureCollection(
        monthly_dates(start_override=smap_start).map(per_month)
    ).flatten()


# ----------------------------
# Main: start three tiny CSV exports
# ----------------------------
if __name__ == "__main__":
    export_table(fc_monthly_ndvi(),      "EA_admin1_monthly_NDVI")
    export_table(fc_monthly_chirps(),    "EA_admin1_monthly_CHIRPS")
    export_table(fc_monthly_smap_rzsm(), "EA_admin1_monthly_SMAP_RZSM")
    print("✅ Three Drive exports started. Check Earth Engine Code Editor → Tasks.")
