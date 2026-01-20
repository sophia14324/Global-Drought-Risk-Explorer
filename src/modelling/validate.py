"""
validate.py
~~~~~~~~~~~

This module provides functions to validate the drought risk index
against external ground‑truth datasets. Typical ground truth sources
include the Standardised Precipitation Evapotranspiration Index (SPEI),
the Palmer Drought Severity Index (PDSI) or reports of drought
disasters. The validation functions compute correlation coefficients
and error metrics between the modelled risk index and the ground truth.

Users are expected to supply preprocessed ground truth data as a
``pandas.DataFrame`` with columns matching those of the indicators
table (e.g., ``ADM0_NAME``, ``ADM1_NAME``, ``date``). See the
companion notebook (e.g. `notebooks/validate_index.ipynb`) for guidance.
"""

from __future__ import annotations

import logging
from typing import Sequence, Tuple

import numpy as np
import pandas as pd

try:
    # Optional dependency: SciPy provides correlation functions
    from scipy.stats import pearsonr, spearmanr  # type: ignore
except ImportError:
    pearsonr = spearmanr = None

logger = logging.getLogger(__name__)


def validate_against_truth(
    df_model: pd.DataFrame,
    df_truth: pd.DataFrame,
    on: Sequence[str] = ("ADM0_NAME", "ADM1_NAME", "date"),
    metric: str = "pearson",
) -> Tuple[float, float]:
    """
    Compute correlation and root mean square error between model and truth.

    Parameters
    ----------
    df_model : pandas.DataFrame
        DataFrame containing the modelled risk index with a column named ``risk_index``.
    df_truth : pandas.DataFrame
        DataFrame containing ground‑truth drought severity values with a column named ``truth_index``.
    on : sequence of str, optional
        Columns to join on. Default joins on ``ADM0_NAME``, ``ADM1_NAME`` and ``date``.
    metric : {"pearson", "spearman"}, optional
        Correlation metric to compute.

    Returns
    -------
    (corr, rmse) : tuple of floats
        The correlation coefficient and root mean square error. If no valid pairs
        are available, both values are NaN.
    """
    merged = pd.merge(
        df_model[list(on) + ["risk_index"]],
        df_truth[list(on) + ["truth_index"]],
        on=list(on),
        how="inner",
    ).dropna(subset=["risk_index", "truth_index"])

    if merged.empty:
        logger.warning("No overlapping data between model and truth; cannot compute validation metrics.")
        return float("nan"), float("nan")

    x = merged["risk_index"].astype(float).to_numpy()
    y = merged["truth_index"].astype(float).to_numpy()

    # compute correlation
    if metric == "pearson":
        if pearsonr is not None:
            corr, _ = pearsonr(x, y)
        else:
            corr = float(np.corrcoef(x, y)[0, 1]) if np.std(x) and np.std(y) else float("nan")
    elif metric == "spearman":
        if spearmanr is not None:
            corr, _ = spearmanr(x, y)
        else:
            rx = pd.Series(x).rank().to_numpy()
            ry = pd.Series(y).rank().to_numpy()
            corr = float(np.corrcoef(rx, ry)[0, 1])
    else:
        raise ValueError(f"Unknown metric: {metric}")

    # compute RMSE
    rmse = float(np.sqrt(np.nanmean((x - y) ** 2)))
    return float(corr), rmse
