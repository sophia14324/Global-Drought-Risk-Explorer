"""Configuration loader for the Global Drought Risk Explorer.

This module reads configuration values from a YAML file located at the
project root (``config.yaml``).  If the file is missing or fields are
absent, sensible defaults are used.  The configuration provides
geographic boundaries, temporal ranges, data directories and weights
for computing the drought risk index.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Any
import os
import yaml

# Locate the project root relative to this file: two levels up from src.
ROOT: Path = Path(__file__).resolve().parents[1]

def _load_yaml(path: Path) -> Dict[str, Any]:
    """Load a YAML configuration file if it exists."""
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as fh:
        try:
            return yaml.safe_load(fh) or {}
        except Exception:
            return {}

_CONFIG_PATH: Path = ROOT / "config.yaml"
_CONFIG: Dict[str, Any] = _load_yaml(_CONFIG_PATH)

# Project directories
DATA_DIR: Path = ROOT / "data"
ASSETS_DIR: Path = ROOT / "assets"
DATA_DIR.mkdir(parents=True, exist_ok=True)
ASSETS_DIR.mkdir(parents=True, exist_ok=True)

def _get(name: str, default: Any) -> Any:
    return _CONFIG.get(name, default)

# Geographic bounds (min_lon, min_lat, max_lon, max_lat)
BBOX = tuple(_get("bbox", [21.8, -11.0, 51.5, 18.0]))
CRS: str = str(_get("crs", "EPSG:4326"))
GRID_DEG: float = float(_get("grid_deg", 0.1))

# Temporal range
START: str = str(_get("start_date", "2001-01-01"))
END: str = str(_get("end_date", "2025-06-30"))

# Google Drive folder for Earth Engine exports
DRIVE_FOLDER: str = str(_get("drive_folder", "gee_monthly"))

# Earth Engine project ID (overrides environment if provided)
EE_PROJECT: str = str(os.getenv("EE_PROJECT") or _get("earth_engine_project", ""))

# Weights for the drought risk index
WEIGHTS: Dict[str, float] = _get("weights", {
    "ndvi_stress": 1.0,
    "rain_deficit": 1.0,
    "soil_dryness": 1.0,
    "temp_anomaly": 1.0,
})

def normalised_weights(weights: Dict[str, float] | None = None) -> Dict[str, float]:
    """Return a copy of the supplied weights normalised to sum to 1.0."""
    w = dict(weights or WEIGHTS)
    total = sum(float(v) for v in w.values()) or 1.0
    return {k: float(v) / total for k, v in w.items()}

__all__ = [
    "ROOT",
    "DATA_DIR",
    "ASSETS_DIR",
    "BBOX",
    "CRS",
    "GRID_DEG",
    "START",
    "END",
    "DRIVE_FOLDER",
    "EE_PROJECT",
    "WEIGHTS",
    "normalised_weights",
]
