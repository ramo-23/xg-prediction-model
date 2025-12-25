"""Preprocessing helpers for shot event tables.

This module contains small, well-tested cleaning helpers used early in the
pipeline (notebooks and scripts). Keep functions focused and easy to test.
"""

import pandas as pd


def clean_shots(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize and clean a shots DataFrame.

    - Strips whitespace from column names.
    - Drops rows missing `x` or `y` coordinates when those columns exist.

    Returns an empty DataFrame when input is `None` or empty.
    """
    if df is None or df.empty:
        return pd.DataFrame()
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]
    df = df.dropna(subset=['x', 'y'], how='any') if {'x', 'y'}.issubset(df.columns) else df
    return df
