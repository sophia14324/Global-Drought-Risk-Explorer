import json
import ee, geemap
from src.config import REGION_SHAPE, START, END

# Load GeoJSON file properly
with open(REGION_SHAPE, "r") as f:
    geojson_data = json.load(f)

REGION_SHAPE = "assets/east_africa_bbox.geojson"

ee.Initialize(project='operating-axis-468409-s4')  # Move after REGION if needed

def get_collection(asset: str, band: str, scale: int = 1000):
    coll = (ee.ImageCollection(asset)
              .filterDate(START, END)
              .filterBounds(REGION)
              .select(band))
    return coll

def export_monthly_mean(asset, band, out_dir, scale=1000):
    coll = get_collection(asset, band, scale)
    months = ee.List.sequence(
        0,
        ee.Number.parse(ee.Date(END).format("YYYY"))
        .subtract(ee.Number.parse(ee.Date(START).format("YYYY")))
        .multiply(12)
    )

    def by_month(m):
        im = (
            coll.filterDate(
                ee.Date(START).advance(m, "month"),
                ee.Date(START).advance(m + 1, "month")
            )
            .mean()
            .set("system:time_start", ee.Date(START).advance(m, "month").millis())
        )
        return im

    monthly = ee.ImageCollection(months.map(by_month))
    geemap.ee_export_image_collection_to_drive(
        monthly,
        folder="gee_monthly",
        region=REGION.geometry(),
        scale=scale,
        crs="EPSG:4326"
    )
