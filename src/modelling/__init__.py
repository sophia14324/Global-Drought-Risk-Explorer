"""Initialize the modelling package.

This file makes the `src.modelling` directory a Python package.  It
exposes key functions for computing risk indices and clustering.
"""
'''
from .risk_index import compute_risk_index, derive_weights, INDICATOR_COLUMNS  # noqa: F401
from .clustering import spatial_cluster, label_category  # noqa: F401
from .validate import validate_against_truth  # noqa: F401

__all__ = [
    "compute_risk_index",
    "derive_weights",
    "INDICATOR_COLUMNS",
    "spatial_cluster",
    "label_category",
    "validate_against_truth",
]
'''
"""Modelling subpackage."""
__all__ = []
