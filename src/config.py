from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
REGION_SHAPE = "assets\east_africa_bbox.geojson"  
BBOX = [21.8, -11.0, 51.5, 18.0]  # lon/lat bounding box of East Africa
CRS = "EPSG:4326"
GRID_DEG = 0.1        # ≈ 0.1° target grid
START = "2001-01-01"
END   = "2025-06-30"
