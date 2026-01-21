"""Utilities for downloading ERA5-Land data."""

from __future__ import annotations

import cdsapi  # type: ignore
import datetime as dt
import logging
from pathlib import Path
from typing import Iterable, Tuple

# Import configuration values from src.config
from src.config import DATA_DIR, START, END, BBOX  # type: ignore

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def _request(variable: str, year: int, target: str) -> None:
    """Submit a retrieval request to the CDS API for ERA5-Land."""
    client = cdsapi.Client()
    request = {
        "product_type": "reanalysis",
        "variable": variable,
        "year": str(year),
        "month": [f"{m:02d}" for m in range(1, 13)],
        "day": [f"{d:02d}" for d in range(1, 32)],
        "time": ["00:00"],
        "format": "netcdf",
        # N, W, S, E
        "area": [BBOX[3], BBOX[0], BBOX[1], BBOX[2]],
    }
    attempts = 0
    while attempts < 3:
        try:
            logger.info(f"Requesting ERA5-Land {variable} {year}")
            client.retrieve("reanalysis-era5-land", request, target)
            logger.info(f"Downloaded {target}")
            return
        except Exception as exc:
            attempts += 1
            logger.warning(f"Attempt {attempts} failed: {exc}")
            if attempts >= 3:
                logger.error(f"Failed to download {variable} {year} after {attempts} attempts.")
                raise

def download(
    years: Iterable[int] | None = None,
    variables: Tuple[str, ...] = (
        "total_precipitation",
        "2m_temperature",
        "volumetric_soil_water_layer_1",
    ),
) -> None:
    """Download ERA5-Land data for the given years and variables."""
    if years is None:
        start_year = dt.datetime.fromisoformat(str(START)).year
        end_year = dt.datetime.fromisoformat(str(END)).year
        years = range(start_year, end_year + 1)
    for year in years:
        for variable in variables:
            target = DATA_DIR / f"era5_{variable}_{year}.nc"
            _request(variable, year, str(target))
