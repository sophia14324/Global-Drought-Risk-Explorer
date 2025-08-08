import xarray as xr, pandas as pd, tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from src.config import DATA_DIR

def lstm_forecast(zone_id, lookback=12, horizon=3):
    risk = xr.open_dataset(DATA_DIR / "risk_index.nc").risk_index
    series = (risk
              .sel(zone=zone_id)   
              .to_pandas()
              .dropna())
    scaler = MinMaxScaler()
    y = scaler.fit_transform(series.values.reshape(-1, 1)).flatten()
    X, Y = [], []
    for i in range(len(y) - lookback - horizon):
        X.append(y[i:i+lookback])
        Y.append(y[i+lookback:i+lookback+horizon])
    X, Y = np.array(X), np.array(Y)
    X = X[..., None]  
    model = tf.keras.Sequential([
        tf.keras.layers.LSTM(64, input_shape=X.shape[1:]),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(horizon)
    ])
    model.compile(loss="mse", optimizer="adam")
    model.fit(X, Y, epochs=100, verbose=0)
    pred = model.predict(X[-1:])[0]
    return scaler.inverse_transform(pred.reshape(-1, 1)).flatten()

'''
from prophet import Prophet

def prophet_forecast(zone_id, periods=6):
    risk = xr.open_dataset(DATA_DIR / "risk_index.nc").risk_index
    df = risk.sel(zone=zone_id).to_dataframe().reset_index()
    df.columns = ["ds", "y"]
    m = Prophet()
    m.fit(df)
    future = m.make_future_dataframe(periods=periods, freq="M")
    fcst  = m.predict(future)
    return fcst[["ds", "yhat", "yhat_lower", "yhat_upper"]]

'''