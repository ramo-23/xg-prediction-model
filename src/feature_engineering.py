"""Feature engineering for xG model."""
from typing import Optional, List
import numpy as np
import pandas as pd


def add_basic_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add distance and angle features given x/y shot coordinates.
    Assumes pitch coordinates where goal is at (0, 0) or similar — adapt as needed.
    """
    if df is None or df.empty:
        return pd.DataFrame()
    df = df.copy()
    if {'x','y'}.issubset(df.columns):
        df['distance'] = np.hypot(df['x'], df['y'])
        df['angle'] = np.arctan2(np.abs(df['y']), df['x'])
    return df


def fe_body_part_onehot(df: pd.DataFrame, col: str = 'body_part') -> pd.DataFrame:
    """One-hot encode body part into `body_foot`, `body_head`, `body_other`.

    Handles common labels: 'Right Foot', 'Left Foot', 'Head', 'Other', etc.
    """
    df = df.copy()
    mapping = {
        'head': 'body_head',
        'right foot': 'body_foot',
        'left foot': 'body_foot',
        'other': 'body_other',
    }
    lname = df[col].fillna('').str.lower()
    df['body_head'] = (lname == 'head').astype(int)
    df['body_foot'] = lname.isin(['right foot', 'left foot']).astype(int)
    df['body_other'] = (~(lname.isin(['right foot', 'left foot', 'head'])) & (lname != '')).astype(int)
    return df


def fe_shot_type(df: pd.DataFrame, notes_col: str = 'notes') -> pd.DataFrame:
    """Infer shot type from `notes` (e.g., 'Free kick', 'Corner', 'Penalty', 'Open play').

    Adds a `shot_type` categorical column.
    """
    df = df.copy()
    def infer(note):
        if pd.isna(note):
            return 'open_play'
        n = str(note).lower()
        if 'free' in n:
            return 'free_kick'
        if 'corner' in n:
            return 'corner'
        if 'pen' in n or 'penalty' in n:
            return 'penalty'
        if 'own' in n:
            return 'own_goal'
        return 'open_play'

    df['shot_type'] = df[notes_col].apply(infer) if notes_col in df.columns else 'open_play'
    return df


def fe_assist_type(df: pd.DataFrame, sca_cols: Optional[List[str]] = None) -> pd.DataFrame:
    """Extract assist type from SCA columns (secondary chance assist). Expects columns like
    'SCA 1' (player) and 'SCA 1.1' (event) based on exported CSV layout. Returns `assist_type`.
    """
    df = df.copy()
    if sca_cols is None:
        # common pattern in FBref export
        sca_cols = [c for c in df.columns if 'SCA' in str(c) and 'event' in str(c).lower()]

    assist = []
    for _, row in df.iterrows():
        found = 'unknown'
        for c in sca_cols:
            val = row.get(c, '')
            if pd.isna(val) or str(val).strip() == '':
                continue
            v = str(val).lower()
            if 'pass' in v:
                found = 'pass'
                break
            if 'cross' in v:
                found = 'cross'
                break
            if 'shot' in v or 'take-on' in v or 'dribble' in v:
                found = 'secondary_shot'
                break
        assist.append(found)
    df['assist_type'] = assist
    return df


def fe_game_half(df: pd.DataFrame, minute_col: str = 'minute') -> pd.DataFrame:
    """Create `minute` numeric where possible and `half` (1 or 2).

    Handles minute strings like '45+2'.
    """
    df = df.copy()

    def parse_min(m):
        try:
            if pd.isna(m):
                return None
            s = str(m)
            if '+' in s:
                base = s.split('+')[0]
                return int(base)
            return int(float(s))
        except Exception:
            return None

    df['minute_num'] = df[minute_col].apply(parse_min) if minute_col in df.columns else None

    def determine_half(x):
        if pd.isna(x):
            return None
        try:
            return 1 if float(x) <= 45 else 2
        except Exception:
            return None

    df['half'] = df['minute_num'].apply(determine_half)
    return df


def fe_big_chance(df: pd.DataFrame, distance_col: str = 'distance', outcome_col: str = 'outcome') -> pd.DataFrame:
    """Flag probable 'big chance' heuristically: very short distance or outcomes containing 'tap' or 'one-on-one'.
    This is a heuristic when explicit annotation isn't available.
    """
    df = df.copy()
    d = df[distance_col] if distance_col in df.columns else pd.Series([None]*len(df))
    df['big_chance'] = ((pd.to_numeric(d, errors='coerce') <= 6)).fillna(False)
    # additional keyword-based heuristics: 'tap' and variants of one-on-one
    outcome_text = df[outcome_col].fillna('').str.lower()
    keyword_flag = outcome_text.str.contains('tap', regex=False).fillna(False) | outcome_text.str.contains('one-on', regex=False).fillna(False) | outcome_text.str.contains('one on', regex=False).fillna(False)
    df['big_chance'] = (df['big_chance'] | keyword_flag).astype(int)
    return df


def calculate_shot_angle(x, y, goal_center_x=105, goal_y1=36.66, goal_y2=43.34):
    """Calculate shot angle in degrees given coordinates.

    Example pitch coordinate convention expected by the caller. Returns NaN if inputs invalid.
    """
    try:
        a1 = np.arctan2(goal_y1 - y, goal_center_x - x)
        a2 = np.arctan2(goal_y2 - y, goal_center_x - x)
        return abs(np.degrees(a1 - a2))
    except Exception:
        return np.nan

