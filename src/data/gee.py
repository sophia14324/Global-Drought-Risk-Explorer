import json
import logging
from pathlib import Path
from typing import Optional

import ee  # type: ignore

# Pull configuration values from src.config.  EE_PROJECT is defined in config.py.
from src.config import START, END, BBOX, CRS, DRIVE_FOLDER, EE_PROJECT  # type: ignore

logger = logging.getLogger(__name__)

def initialise_ee(project: Optional[str] = None) -> None:
    """
    Initialise the Earth Engine client using a project ID from the
    configuration or the provided argument.

    This function wraps `ee.Initialize` in a try/except block and logs
    any errors encountered during initialisation.
    """
    project_id = project or EE_PROJECT or None
    try:
        ee.Initialize(project=project_id)
        logger.info("Initialised Earth Engine with project %s", project_id)
    except Exception as exc:
        logger.error("Failed to initialise Earth Engine: %s", exc)
        raise

def load_region_geometry() -> ee.Geometry:
    """
    Return an ee.Geometry for the region of interest.

    The function tries the following, in order:
      1. Load a GeoJSON file from the `assets` directory (if present).
      2. Union together GAUL Level‑0 country boundaries for East Africa.
      3. Fall back to a simple bounding box defined in the configuration.
    """
    # Attempt to read a local GeoJSON file if it exists.  The file is expected
    # at assets/east_africa_bbox.geojson relative to the project root.
    region_path = Path(__file__).resolve().parents[2] / "assets" / "east_africa_bbox.geojson"
    if region_path.exists():
        try:
            import geemap  # type: ignore
            gj = json.loads(region_path.read_text(encoding="utf-8"))
            ee_obj = geemap.geojson_to_ee(gj)
            if isinstance(ee_obj, ee.geometry.Geometry):
                return ee_obj
            if isinstance(ee_obj, ee.feature.Feature):
                return ee_obj.geometry()
            if isinstance(ee_obj, ee.featurecollection.FeatureCollection):
                return ee_obj.geometry()
        except Exception as exc:
            logger.warning("Failed to load GeoJSON region: %s", exc)

    # Fallback: union of East African countries from GAUL level 0.
    try:
        EA = [
            "Kenya", "Uganda", "Tanzania", "Ethiopia", "Somalia",
            "South Sudan", "Rwanda", "Burundi", "Sudan", "Eritrea", "Djibouti"
        ]
        gaul0 = ee.FeatureCollection("FAO/GAUL/2015/level0")
        return gaul0.filter(ee.Filter.inList("ADM0_NAME", EA)).geometry()
    except Exception as exc:
        logger.warning("Failed to load GAUL boundaries: %s", exc)

    # Final fallback: use the bounding box from config
    return ee.Geometry.BBox(*BBOX)

# Initialise Earth Engine at module import time, using the configuration's project ID.
initialise_ee()

# Construct region geometry and administrative boundaries
REGION = load_region_geometry()
GAUL1 = ee.FeatureCollection("FAO/GAUL/2015/level1").map(
    lambda f: f.intersection(REGION, 1)
).filterBounds(REGION)

def monthly_dates(start_override: Optional[str] = None, end_override: Optional[str] = None) -> ee.List:
    """
    Generate a sequence of monthly start dates between START and END (inclusive).

    Parameters
    ----------
    start_override : str, optional
        Override the default start date from the configuration (ISO format).
    end_override : str, optional
        Override the default end date from the configuration (ISO format).

    Returns
    -------
    ee.List
        A list of `ee.Date` objects representing the first day of each month.
    """
    start = ee.Date(start_override or START)
    end = ee.Date(end_override or END)
    n = end.difference(start, "month").floor()
    return ee.List.sequence(0, n.subtract(1)).map(lambda m: start.advance(m, "month"))

def export_table(feature_collection: ee.FeatureCollection, description: str, folder: str = DRIVE_FOLDER) -> None:
    """
    Export a FeatureCollection to Google Drive as a CSV.

    The function starts the export task asynchronously and logs a message
    indicating the destination folder and description.
    """
    task = ee.batch.Export.table.toDrive(
        collection=feature_collection,
        description=description,
        folder=folder,
        fileFormat="CSV",
        selectors=["ADM0_NAME", "ADM1_NAME", "date", "variable", "value"],
    )
    task.start()
    logger.info("▶️ Started export: %s → Drive/%s", description, folder)

def fc_monthly_ndvi() -> ee.FeatureCollection:
    """
    Compute monthly mean NDVI for each admin‑1 unit.

    Returns
    -------
    ee.FeatureCollection
        A flattened collection with properties ADM0_NAME, ADM1_NAME, date,
        variable (NDVI) and value (mean NDVI).
    """
    coll = ee.ImageCollection("MODIS/061/MOD13A3").select("NDVI")
    def per_month(d):
        d = ee.Date(d)
        img = coll.filterDate(d, d.advance(1, "month")).mean().clip(REGION)
        ndvi = img.multiply(0.0001)  # scale factor
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

def fc_monthly_chirps() -> ee.FeatureCollection:
    """
    Compute monthly total rainfall from CHIRPS for each admin‑1 unit.

    Returns
    -------
    ee.FeatureCollection
        A flattened collection with variable CHIRPS_mm and mean rainfall.
    """
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

def fc_monthly_smap_rzsm() -> ee.FeatureCollection:
    """
    Compute monthly root‑zone soil moisture (SMAP RZSM) for each admin‑1 unit.

    Returns
    -------
    ee.FeatureCollection
        A flattened collection with variable SMAP_RZSM and mean soil moisture.
    """
    coll = ee.ImageCollection("NASA/SMAP/SPL4SMGP/008").select("sm_rootzone")
    smap_start = "2015-04-01"  # first full month with SMAP data
    def per_month(d):
        d = ee.Date(d)
        month_coll = coll.filterDate(d, d.advance(1, "month"))
        img = ee.Image(ee.Algorithms.If(
            month_coll.size().gt(0),
            month_coll.mean().rename("sm_rootzone"),
            ee.Image.constant(float("nan")).rename("sm_rootzone")
        )).clip(REGION)
        stats = img.reduceRegions(
            collection=GAUL1,
            reducer=ee.Reducer.mean(),
            scale=11000,  # native resolution (~9–11 km)
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

if __name__ == "__main__":
    # Kick off exports when run as a script
    export_table(fc_monthly_ndvi(),      "EA_admin1_monthly_NDVI")
    export_table(fc_monthly_chirps(),    "EA_admin1_monthly_CHIRPS")
    export_table(fc_monthly_smap_rzsm(), "EA_admin1_monthly_SMAP_RZSM")
    print("✅ Three Drive exports started. Check Earth Engine Code Editor → Tasks.")
