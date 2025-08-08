# ERA5-Land (precip, temp, soil m)
import cdsapi, xarray as xr, datetime as dt
from src.config import DATA_DIR, START, END, BBOX

def _request(variable: str, year: int, target: str):
    client = cdsapi.Client()
    client.retrieve(
        "reanalysis-era5-land",             
        {
            "variable": variable,
            "year": str(year),
            "month": [f"{m:02d}" for m in range(1, 13)],
            "day":   [f"{d:02d}" for d in range(1, 32)],
            "time":  ["00:00"],
            "format": "netcdf",
            "area":  [BBOX[3], BBOX[0], BBOX[1], BBOX[2]],  # N,W,S,E
        },
        target,
    )

def download(years=range(2001, 2026), variables=("total_precipitation", 
                                                 "2m_temperature",
                                                 "volumetric_soil_water_layer_1")):
    DATA_DIR.mkdir(exist_ok=True, parents=True)
    for var in variables:
        for yr in years:
            fn = DATA_DIR / f"era5_{var}_{yr}.nc"
            if not fn.exists():
                _request(var, yr, str(fn))


if __name__ == "__main__":                      
    download(years=[2020])
    print("🎉 ERA5 download task(s) submitted. Check data folder.")
