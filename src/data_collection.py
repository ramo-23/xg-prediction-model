"""Data collection utilities (FBref / soccerdata wrappers).

This module centralises functions that fetch data from FBref via the
`soccerdata` package when available. Functions return empty DataFrames as a
safe default if `soccerdata` is not installed or network access fails.

Provided helpers:
- `fetch_shot_events`, `fetch_match_schedule`, `fetch_team_stats`,
  `fetch_player_stats`: thin wrappers that prefer robust failure modes.
- `save_raw_data`: persist DataFrame to CSV and write a small README with
  basic metadata.

The code is intentionally defensive to make notebooks and scripts runnable in
offline or CI environments where `soccerdata` may not be present.
"""

from typing import Optional

try:
    from soccerdata import FBref
except Exception:
    FBref = None

import pandas as pd
import os
from datetime import datetime


def _safe_call(fbref_obj, method_name: str) -> pd.DataFrame:
    """Call a FBref method if it exists, otherwise return empty DataFrame."""
    if fbref_obj is None:
        return pd.DataFrame()
    fn = getattr(fbref_obj, method_name, None)
    if fn is None:
        return pd.DataFrame()
    try:
        return fn()
    except Exception:
        return pd.DataFrame()


def fetch_shot_events(leagues: str, seasons: list) -> pd.DataFrame:
    """Fetch shot events via soccerdata.FBref.

    Returns an empty DataFrame if `soccerdata` is not available.
    """
    if FBref is None:
        return pd.DataFrame()

    fbref = FBref(leagues=leagues, seasons=seasons)
    shots = fbref.read_shot_events()
    return shots


def fetch_match_schedule(leagues: str, seasons: list) -> pd.DataFrame:
    """Fetch match schedule / results using FBref wrapper.

    Returns empty DataFrame if unavailable.
    """
    if FBref is None:
        return pd.DataFrame()
    fbref = FBref(leagues=leagues, seasons=seasons)
    return _safe_call(fbref, "read_match_results")


def fetch_team_stats(leagues: str, seasons: list) -> pd.DataFrame:
    """Fetch aggregated team statistics.

    Returns empty DataFrame if unavailable.
    """
    if FBref is None:
        return pd.DataFrame()
    fbref = FBref(leagues=leagues, seasons=seasons)
    # try common team stat method names
    for name in ("read_team_stats", "read_teams", "read_team_data"):
        df = _safe_call(fbref, name)
        if not df.empty:
            return df
    return pd.DataFrame()


def fetch_player_stats(leagues: str, seasons: list) -> pd.DataFrame:
    """Fetch player statistics (may be optional / large).

    Returns empty DataFrame if unavailable.
    """
    if FBref is None:
        return pd.DataFrame()
    fbref = FBref(leagues=leagues, seasons=seasons)
    for name in ("read_player_stats", "read_players", "read_squad"):
        df = _safe_call(fbref, name)
        if not df.empty:
            return df
    return pd.DataFrame()


def save_raw_data(df: pd.DataFrame, name: str, description: Optional[str] = None, out_dir: str = "data/processed") -> str:
    """Save DataFrame to CSV in `out_dir/raw/` and write a small README with metadata.

    Returns the path to the saved CSV.
    """
    os.makedirs(out_dir, exist_ok=True)
    timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    filename = f"{name}_{timestamp}.csv"
    csv_path = os.path.join(out_dir, filename)
    df.to_csv(csv_path, index=False)

    # write a small README describing the raw file
    readme_path = os.path.join(out_dir, f"{name}_{timestamp}_README.md")
    with open(readme_path, "w", encoding="utf-8") as fh:
        fh.write(f"# Raw data: {name}\n\n")
        fh.write(f"created_utc: {timestamp}\n\n")
        if description:
            fh.write(f"description: {description}\n\n")
        fh.write(f"rows: {len(df)}\n")
        fh.write(f"columns: {len(df.columns)}\n\n")
        fh.write("columns:\n")
        for col in df.columns:
            fh.write(f"- {col}\n")

    return csv_path
