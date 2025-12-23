"""Simple EDA for shot events CSVs.

- Auto-detects latest `shots_events_*.csv` in `data/raw/`.
- Fixes duplicated header-row produced by exported CSVs.
- Produces:
  - `shots_outcome_counts.png` (outcome distribution)
  - `xg_histogram.png` (xG distribution)
  - `distance_vs_xg.png` (scatter distance vs xG)
  - `bodypart_counts.png` (body part usage)

Run with venv active from repo root:
    python scripts/eda_shots.py

"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import os
from pathlib import Path

RAW_DIR = Path("data/raw")
OUT_DIR = Path("results/figures")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def find_latest_shots_csv():
    files = sorted(RAW_DIR.glob("shots_events_*.csv"), key=os.path.getmtime)
    if not files:
        raise FileNotFoundError("No shots_events_*.csv files in data/raw/")
    return files[-1]


def read_and_fix_csv(path: Path) -> pd.DataFrame:
    # read first two rows to detect duplicated header
    df_head = pd.read_csv(path, header=None, nrows=2)
    first_row = df_head.iloc[0].fillna("").astype(str)
    second_row = df_head.iloc[1].fillna("").astype(str)

    # If first row mostly empty and second row contains column-like values, use header=1
    num_nonempty_first = (first_row.str.strip() != "").sum()
    num_nonempty_second = (second_row.str.strip() != "").sum()

    if num_nonempty_first <= 2 and num_nonempty_second > num_nonempty_first:
        df = pd.read_csv(path, header=1)
    else:
        df = pd.read_csv(path, header=0)

    # strip column names
    df.columns = [c.strip() if isinstance(c, str) else c for c in df.columns]
    return df


def plot_outcome_counts(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(6,4))
    sns.countplot(y=df['outcome'].fillna('Unknown'), order=df['outcome'].value_counts().index, ax=ax)
    ax.set_title('Shot outcomes')
    plt.tight_layout()
    out = OUT_DIR / 'shots_outcome_counts.png'
    fig.savefig(out)
    plt.close(fig)
    print('Saved', out)


def plot_xg_hist(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(6,4))
    sns.histplot(x=df['xG'].dropna().astype(float), bins=30, kde=True, ax=ax)
    ax.set_title('xG distribution')
    plt.tight_layout()
    out = OUT_DIR / 'xg_histogram.png'
    fig.savefig(out)
    plt.close(fig)
    print('Saved', out)


def plot_distance_vs_xg(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(6,4))
    sns.scatterplot(x=df['distance'].astype(float), y=df['xG'].astype(float), alpha=0.6, ax=ax)
    ax.set_xlabel('Distance')
    ax.set_ylabel('xG')
    ax.set_title('Distance vs xG')
    plt.tight_layout()
    out = OUT_DIR / 'distance_vs_xg.png'
    fig.savefig(out)
    plt.close(fig)
    print('Saved', out)


def plot_bodypart_counts(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(6,4))
    sns.countplot(y=df['body_part'].fillna('Unknown'), order=df['body_part'].value_counts().index, ax=ax)
    ax.set_title('Body part usage')
    plt.tight_layout()
    out = OUT_DIR / 'bodypart_counts.png'
    fig.savefig(out)
    plt.close(fig)
    print('Saved', out)


def main():
    path = find_latest_shots_csv()
    print('Loading', path)
    df = read_and_fix_csv(path)
    # quick coercions
    for col in ['xG','distance']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # compute simple summary
    total_shots = len(df)
    goals = (df['outcome'].str.lower() == 'goal').sum() if 'outcome' in df.columns else None
    print(f'Total shots: {total_shots}, goals: {goals}')

    # plots
    if 'outcome' in df.columns:
        plot_outcome_counts(df)
    if 'xG' in df.columns:
        plot_xg_hist(df)
    if 'distance' in df.columns and 'xG' in df.columns:
        plot_distance_vs_xg(df)
    if 'body_part' in df.columns:
        plot_bodypart_counts(df)

    print('EDA complete. Figures written to', OUT_DIR)


if __name__ == '__main__':
    main()
