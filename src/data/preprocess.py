import xarray as xr, pandas as pd, numpy as np, rioxarray as rio
from src.config import DATA_DIR, GRID_DEG

def monthly_mean(ds: xr.Dataset, var: str):
    return (ds[var]
            .resample(time="1M")
            .mean())

def to_grid(ds):
    return (ds
            .rio.reproject("EPSG:4326")   # safety
            .rio.reproject(
                ds.rio.crs,
                resolution=GRID_DEG,
                nodata=np.nan))

def zscore(da):
    return (da - da.groupby("time.month").mean()) / da.groupby("time.month").std()

def build_indicators():
    pr = xr.open_mfdataset(DATA_DIR / "era5_total_precipitation_*.nc")
    temp = xr.open_mfdataset(DATA_DIR / "era5_2m_temperature_*.nc")
    soil = xr.open_mfdataset(DATA_DIR / "era5_volumetric_soil_water_layer_1_*.nc")
    
    rain_deficit = -zscore(monthly_mean(pr, "tp").pipe(to_grid))
    heat_anom    =  zscore(monthly_mean(temp, "t2m").pipe(to_grid))
    soil_dry     = -zscore(monthly_mean(soil, "swvl1").pipe(to_grid))
    
    ndvi = xr.open_mfdataset(str(DATA_DIR / "gee_monthly/NDVI*.tif"), 
                             combine="nested", concat_dim="time")
    ndvi = ndvi.rename({"band_data": "ndvi"})
    ndvi_stress = -zscore(ndvi.ndvi.pipe(to_grid))  # lower NDVI → stress
    
    indicators = xr.merge([rain_deficit.rename("rain_deficit"),
                           heat_anom.rename("temp_anomaly"),
                           soil_dry.rename("soil_dryness"),
                           ndvi_stress.rename("ndvi_stress")])
    
    indicators = indicators.rolling(time=3, center=True).mean()  # smooth
    indicators.to_netcdf(DATA_DIR / "indicators.nc")
    return indicators
