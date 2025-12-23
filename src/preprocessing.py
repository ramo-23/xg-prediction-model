"""Preprocessing helpers for shot events."""
import pandas as pd


def clean_shots(df: pd.DataFrame) -> pd.DataFrame:
    """Basic cleaning: drop NA and normalize column names."""
    if df is None or df.empty:
        return pd.DataFrame()
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]
    df = df.dropna(subset=['x', 'y'], how='any') if {'x','y'}.issubset(df.columns) else df
    return df
