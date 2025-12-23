"""Phase 1 data collection script.

Usage: run this script from the repo root with the same Python environment
that has `soccerdata` installed.

It collects shot events, match schedule, team stats, and player stats (if available),
and saves each as a timestamped CSV under `data/raw/` with a small README.
"""

import sys
import os

# Ensure project root is on sys.path so `src` package imports work when running
# the script from `scripts/` or other working directories.
repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from src.data_collection import (
    fetch_shot_events,
    fetch_match_schedule,
    fetch_team_stats,
    fetch_player_stats,
    save_raw_data,
)
import argparse
import re


def _normalize_season(s: str) -> str:
    """Normalize season formats like '2025-2026' -> '2025-26'."""
    s = s.strip()
    m = re.match(r"^(\d{4})-(\d{4})$", s)
    if m:
        return f"{m.group(1)}-{m.group(2)[2:]}"
    return s


def main():
    parser = argparse.ArgumentParser(description="Collect Phase 1 soccer data (shots, schedule, team/player stats)")
    parser.add_argument("--league", default="ENG-Premier League", help="FBref league code (default: ENG-Premier League)")
    parser.add_argument("--seasons", default="2025-26", help="Comma-separated seasons, e.g. '2025-26' (default: 2025-26)")
    args = parser.parse_args()

    leagues = args.league
    # seasons can be passed as comma-separated values already in FBref-compatible form
    seasons = [s.strip() for s in args.seasons.split(",") if s.strip()]

    print(f"Collecting data for league={leagues}, seasons={seasons}")

    shots = fetch_shot_events(leagues=leagues, seasons=seasons)
    if not shots.empty:
        path = save_raw_data(shots, "shots_events", description=f"FBref shot events for {leagues}, {', '.join(seasons)}", out_dir="data/raw")
        print("Saved shots to:", path)
    else:
        print("No shot events fetched (soccerdata unavailable or empty).")

    schedule = fetch_match_schedule(leagues=leagues, seasons=seasons)
    if not schedule.empty:
        path = save_raw_data(schedule, "match_schedule", description="Match schedule/results", out_dir="data/raw")
        print("Saved schedule to:", path)
    else:
        print("No match schedule fetched.")

    team_stats = fetch_team_stats(leagues=leagues, seasons=seasons)
    if not team_stats.empty:
        path = save_raw_data(team_stats, "team_stats", description="Team-level aggregate stats", out_dir="data/raw")
        print("Saved team stats to:", path)
    else:
        print("No team stats fetched.")

    player_stats = fetch_player_stats(leagues=leagues, seasons=seasons)
    if not player_stats.empty:
        path = save_raw_data(player_stats, "player_stats", description="Player statistics (may be large)", out_dir="data/raw")
        print("Saved player stats to:", path)
    else:
        print("No player stats fetched.")


if __name__ == "__main__":
    main()
