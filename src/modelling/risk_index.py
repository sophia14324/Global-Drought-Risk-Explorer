import xarray as xr
from src.config import DATA_DIR

def compute():
    ind = xr.open_dataset(DATA_DIR / "indicators.nc")
    risk = (ind.rain_deficit + ind.ndvi_stress +
            ind.soil_dryness + ind.temp_anomaly) / 4
    risk = risk.rename("risk_index")
    risk.to_netcdf(DATA_DIR / "risk_index.nc")
    return risk
