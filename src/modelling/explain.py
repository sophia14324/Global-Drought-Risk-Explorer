import shap, xarray as xr, numpy as np
from src.config import DATA_DIR

def shap_grid():
    ind = xr.open_dataset(DATA_DIR / "indicators.nc")
    risk = xr.open_dataset(DATA_DIR / "risk_index.nc").risk_index
    stacked = ind.to_array().stack(z=("variable", "lat", "lon")).transpose("z", "time")
    X = stacked.values.T  # shape: (time, features)
    y = risk.stack(z=("lat", "lon")).values.T.mean(axis=1)  # average
    explainer = shap.Explainer(shap.sample(X, 500))
    shap_values = explainer(X)
    shap.summary_plot(shap_values, feature_names=stacked.variable.values)
