"""
clustering.py
~~~~~~~~~~~~~

Spatial clustering utilities for drought risk patterns.

This version refactors the original implementation to allow clustering
either purely on the time-series of a drought risk metric (e.g. the
smoothed risk index) or, optionally, using latitude/longitude to take
geographic proximity into account.  It uses scikit‑learn's DBSCAN by
default, with support for Euclidean and haversine distance metrics.

Example usage::

    import pandas as pd
    from src.modelling.clustering import spatial_cluster, label_category

    # Suppose df has columns: ADM0_NAME, ADM1_NAME, date, risk_index_sm3, lat, lon
    clusters = spatial_cluster(
        df,
        value_col="risk_index_sm3",
        lat_col="lat",
        lon_col="lon",
        min_cluster_size=10,
        metric="haversine",
        eps=0.2,  # ~20° (~2,000 km); adjust as needed
    )

    df["cluster"] = clusters
    df["cluster_category"] = label_category(df["cluster"])
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

try:
    from sklearn.cluster import DBSCAN  # type: ignore
except ImportError as exc:
    raise ImportError(
        "scikit-learn is required for clustering. Install via `pip install scikit-learn`."
    ) from exc


def spatial_cluster(
    df: pd.DataFrame,
    value_col: str,
    lat_col: Optional[str] = None,
    lon_col: Optional[str] = None,
    min_cluster_size: int = 25,
    metric: str = "euclidean",
    eps: float = 0.5,
    **kwargs,
) -> pd.Series:
    """
    Cluster administrative units based on drought risk patterns.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing the value column and optionally latitude/longitude columns.
    value_col : str
        Name of the column containing the value to cluster (e.g. a smoothed risk index).
    lat_col, lon_col : str, optional
        Names of the columns containing latitude and longitude in decimal degrees. If provided
        and ``metric='haversine'``, the lat/lon will be converted to radians for geodesic distance.
    min_cluster_size : int, optional
        Minimum number of samples in a cluster (DBSCAN's `min_samples`).
    metric : {"euclidean", "haversine"}, optional
        Distance metric for DBSCAN.  ``"haversine"`` requires lat/lon and converts degrees to radians.
    eps : float, optional
        Maximum distance between two samples for them to be considered in the same neighbourhood.
        When using ``"haversine"``, `eps` should be specified in radians (e.g. 0.1 ≈ 637 km).
    **kwargs : dict
        Additional keyword arguments forwarded to ``DBSCAN``.

    Returns
    -------
    pandas.Series
        A Series of cluster labels (integers).  Noise points are labelled as -1.
    """
    # Flatten the time series into a single vector per region
    group_cols = [c for c in df.columns if c not in {value_col, lat_col, lon_col, "date"}]
    pivot = (
        df.pivot_table(index=group_cols, columns="date", values=value_col, aggfunc="mean")
        .fillna(0.0)
    )
    features = pivot.values

    # Append or substitute lat/lon information if provided
    if lat_col and lon_col:
        coords = df.groupby(group_cols)[[lat_col, lon_col]].first().loc[pivot.index]
        coords_rad = np.radians(coords[[lat_col, lon_col]].to_numpy())
        if metric == "euclidean":
            # Append lat/lon to the feature space
            features = np.hstack([features, coords_rad])
        elif metric == "haversine":
            # Use lat/lon only for haversine metric
            features = coords_rad
        else:
            raise ValueError(f"Unsupported metric: {metric}")

    clustering = DBSCAN(
        eps=eps,
        min_samples=min_cluster_size,
        metric=metric,
        **kwargs,
    )
    labels = clustering.fit_predict(features)
    return pd.Series(labels, index=pivot.index)

# ---------------------------------------------------------------------------
# Label mapping helper
# ---------------------------------------------------------------------------

def label_category(labels: pd.Series | np.ndarray | list) -> pd.Series:
    """Map numeric cluster labels to descriptive categories.

    The mapping is defined as follows:

    * -1 → "Noisy"  (noise/outlier)
    *  0 → "Stable"
    *  1 → "At‑Risk"
    *  2 → "Crisis"
    * any other integer → "Other"
    """
    import pandas as pd  # Local import to avoid unconditional dependency

    mapping = {
        -1: "Noisy",
         0: "Stable",
         1: "At‑Risk",
         2: "Crisis",
    }
    # Convert to Series for uniform processing
    if not isinstance(labels, pd.Series):
        labels = pd.Series(labels)
    return labels.map(lambda x: mapping.get(int(x), "Other"))

__all__.append("label_category")


