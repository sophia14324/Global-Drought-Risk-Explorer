#!/usr/bin/env python
"""
Command-line script to build monthly indicators and compute the drought risk index.
"""

from __future__ import annotations

import argparse
import logging

# Import build_indicators from src.data.preprocess
from src.data.preprocess import build_indicators

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build monthly drought indicators and risk index.")
    parser.add_argument(
        "--weights",
        type=str,
        default=None,
        help=(
            "Optional path to a YAML/JSON file containing custom weights. "
            "The file should map indicator names "
            "(ndvi_stress, rain_deficit, soil_dryness, temp_anomaly) to numeric weights."
        ),
    )
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    weights = None
    if args.weights:
        import yaml  # type: ignore
        with open(args.weights, "r", encoding="utf-8") as fh:
            weights = yaml.safe_load(fh)
    df = build_indicators(weights=weights)
    logging.info("Built indicators table with %d rows", len(df))

if __name__ == "__main__":
    main()
