import hdbscan, numpy as np, xarray as xr, pandas as pd
from sklearn.preprocessing import StandardScaler
from src.config import DATA_DIR, REGION_SHAPE

def spatial_cluster(min_cluster_size=25):
    ds = xr.open_dataset(DATA_DIR / "risk_index.nc")
    da = ds.risk_index

    stacked = da.stack(z=("lat", "lon")).transpose("z", "time")
    X = StandardScaler().fit_transform(stacked.values)
    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, metric="euclidean")
    labels = clusterer.fit_predict(X)

    label_map = (xr.DataArray(labels, dims="z")
                   .unstack("z")
                   .assign_coords(time=da.time.isel(time=0)))
    label_map.rio.write_crs("EPSG:4326", inplace=True)
    label_map.name = "cluster_label"
    label_map.to_netcdf(DATA_DIR / "clusters.nc")
    return label_map

def label_category(labels):
    mapping = {-1: "Noisy", 0: "Stable", 1: "At-Risk", 2: "Crisis"}
    return xr.apply_ufunc(lambda x: mapping.get(x, "Other"),
                          labels, vectorize=True)
