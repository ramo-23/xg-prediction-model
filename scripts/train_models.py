"""Train models for xG prediction.

Steps:
- Loads the latest processed shots CSV from `data/processed/` (created by notebooks).
- Builds feature preprocessing pipeline using available columns.
- Splits into train/val/test (60/20/20) stratified on target (goal vs no goal).
- Trains Logistic Regression (baseline), Random Forest, and XGBoost with simple CV.
- Handles class imbalance using `class_weight` or `scale_pos_weight`.
- Calibrates probabilities using `CalibratedClassifierCV` on validation fold.
- Saves trained models (joblib) and metrics (JSON) under `results/metrics/`.

Run with venv active from repo root:
    python scripts/train_models.py
"""

import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from typing import Literal
from sklearn.calibration import CalibratedClassifierCV
import time

def _make_calibrator(estimator, method: Literal['sigmoid', 'isotonic'] = 'isotonic'):
    """Return a CalibratedClassifierCV instance.

    If `FrozenEstimator` is available (sklearn >= 1.6) use it; otherwise fall back
    to `cv='prefit'` for compatibility. Uses dynamic lookup to avoid static import
    errors from type checkers.
    """
    try:
        mod = __import__('sklearn.calibration', fromlist=['FrozenEstimator'])
        Frozen = getattr(mod, 'FrozenEstimator', None)
    except Exception:
        Frozen = None

    if Frozen is not None:
        return CalibratedClassifierCV(Frozen(estimator), method=method)
    return CalibratedClassifierCV(estimator, cv='prefit', method=method)
from sklearn.metrics import classification_report, roc_auc_score, brier_score_loss
import xgboost as xgb
from sklearn.neural_network import MLPClassifier

RAW_PROC = Path('data/processed')
OUT = Path('results/metrics')
OUT.mkdir(parents=True, exist_ok=True)


def find_latest_processed():
    files = sorted(RAW_PROC.glob('processed_shots_*.csv'), key=lambda p: p.stat().st_mtime)
    if not files:
        raise FileNotFoundError('No processed_shots_*.csv files in data/processed/')
    return files[-1]


def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df


def prepare_features(df: pd.DataFrame):
    # Target: outcome == 'Goal' (case-insensitive)
    df = df.copy()
    if 'outcome' not in df.columns:
        raise ValueError('Input must contain an `outcome` column with goal annotations')
    y = df['outcome'].astype(str).str.lower().eq('goal').astype(int)

    # Candidate numeric features
    numeric_feats = []
    for c in ['distance', 'minute_num']:
        if c in df.columns:
            numeric_feats.append(c)

    # Candidate binary columns added by FE
    binary_feats = [c for c in ['body_head', 'body_foot', 'body_other', 'big_chance', 'half'] if c in df.columns]

    # Categorical features to one-hot
    cat_feats = []
    if 'shot_type' in df.columns:
        cat_feats.append('shot_type')
    if 'assist_type' in df.columns:
        cat_feats.append('assist_type')

    # Optional contextual features (home/away, game state)
    # Prefer explicit binary `is_home` if present; otherwise use `home_away` as categorical
    if 'is_home' in df.columns and 'is_home' not in binary_feats:
        binary_feats.append('is_home')
    if 'home_away' in df.columns and 'home_away' not in cat_feats:
        cat_feats.append('home_away')
    if 'game_state' in df.columns and 'game_state' not in cat_feats:
        cat_feats.append('game_state')
    # Also accept one-hot style game_state_* columns
    for gs in ['Winning', 'Drawing', 'Losing']:
        col = f'game_state_{gs}'
        if col in df.columns and col not in binary_feats:
            binary_feats.append(col)

    feature_cols = numeric_feats + binary_feats + cat_feats
    X = df[feature_cols].copy()
    return X, y, numeric_feats, binary_feats, cat_feats


def build_preprocessor(numeric_feats, binary_feats, cat_feats):
    numeric_pipeline = Pipeline([
        ('impute', SimpleImputer(strategy='median')),
        ('scale', StandardScaler())
    ])
    binary_pipeline = Pipeline([
        ('impute', SimpleImputer(strategy='most_frequent'))
    ])
    cat_pipeline = Pipeline([
        ('impute', SimpleImputer(strategy='constant', fill_value='missing')),
        ('ohe', OneHotEncoder(handle_unknown='ignore'))
    ])
    transformers = []
    if numeric_feats:
        transformers.append(('num', numeric_pipeline, numeric_feats))
    if binary_feats:
        transformers.append(('bin', binary_pipeline, binary_feats))
    if cat_feats:
        transformers.append(('cat', cat_pipeline, cat_feats))
    preprocessor = ColumnTransformer(transformers, remainder='drop')
    return preprocessor


def train_and_evaluate(X, y, preprocessor):
    # Split 60/40 first, then split 40 into 20/20
    X_train_full, X_hold, y_train_full, y_hold = train_test_split(X, y, test_size=0.4, stratify=y, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_hold, y_hold, test_size=0.5, stratify=y_hold, random_state=42)

    results = {}

    # Logistic Regression baseline
    pipe_lr = Pipeline([
        ('pre', preprocessor),
        ('clf', LogisticRegression(class_weight='balanced', max_iter=1000))
    ])
    start = time.time()
    pipe_lr.fit(X_train_full, y_train_full)
    # calibrate on validation
    calib_lr = _make_calibrator(pipe_lr, method='isotonic')
    calib_lr.fit(X_val, y_val)
    lr_time = time.time() - start
    y_pred_proba = calib_lr.predict_proba(X_test)[:, 1]
    y_pred = (y_pred_proba >= 0.5).astype(int)
    results['logistic'] = {**summarize_metrics(y_test, y_pred, y_pred_proba), 'train_time_seconds': lr_time}
    joblib.dump(calib_lr, OUT / 'model_logistic_calibrated.joblib')

    # Random Forest with simple grid search
    rf = RandomForestClassifier(class_weight='balanced', n_jobs=-1, random_state=42)
    pipe_rf = Pipeline([('pre', preprocessor), ('clf', rf)])
    param_grid = {'clf__n_estimators': [100, 200], 'clf__max_depth': [None, 10]}
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    gs_rf = GridSearchCV(pipe_rf, param_grid, cv=cv, scoring='roc_auc', n_jobs=-1)
    start = time.time()
    gs_rf.fit(X_train_full, y_train_full)
    best_rf = gs_rf.best_estimator_
    calib_rf = _make_calibrator(best_rf, method='isotonic')
    calib_rf.fit(X_val, y_val)
    rf_time = time.time() - start
    y_pred_proba = calib_rf.predict_proba(X_test)[:, 1]
    y_pred = (y_pred_proba >= 0.5).astype(int)
    results['random_forest'] = {'best_params': gs_rf.best_params_, **summarize_metrics(y_test, y_pred, y_pred_proba), 'train_time_seconds': rf_time}
    joblib.dump(calib_rf, OUT / 'model_random_forest_calibrated.joblib')

    # XGBoost with simple tuning
    xgb_clf = xgb.XGBClassifier(eval_metric='logloss', n_jobs=-1, random_state=42)
    pipe_xgb = Pipeline([('pre', preprocessor), ('clf', xgb_clf)])
    param_grid = {'clf__n_estimators': [100, 200], 'clf__max_depth': [3, 6]}
    gs_xgb = GridSearchCV(pipe_xgb, param_grid, cv=cv, scoring='roc_auc', n_jobs=-1)
    # estimate scale_pos_weight for imbalance
    pos = y_train_full.sum()
    neg = len(y_train_full) - pos
    scale_pos_weight = max(1, neg / max(1, pos))
    xgb_clf.set_params(scale_pos_weight=scale_pos_weight)
    start = time.time()
    gs_xgb.fit(X_train_full, y_train_full)
    best_xgb = gs_xgb.best_estimator_
    calib_xgb = _make_calibrator(best_xgb, method='isotonic')
    calib_xgb.fit(X_val, y_val)
    xgb_time = time.time() - start
    y_pred_proba = calib_xgb.predict_proba(X_test)[:, 1]
    y_pred = (y_pred_proba >= 0.5).astype(int)
    results['xgboost'] = {'best_params': gs_xgb.best_params_, **summarize_metrics(y_test, y_pred, y_pred_proba), 'train_time_seconds': xgb_time}
    joblib.dump(calib_xgb, OUT / 'model_xgboost_calibrated.joblib')

    # Neural Network (MLP) - optional deep model
    # A simple MLPClassifier to capture potentially complex interactions.
    mlp = MLPClassifier(hidden_layer_sizes=(100,), early_stopping=True,
                        max_iter=500, random_state=42)
    pipe_mlp = Pipeline([('pre', preprocessor), ('clf', mlp)])
    # Fit on full training, calibrate on validation like other models
    start = time.time()
    pipe_mlp.fit(X_train_full, y_train_full)
    calib_mlp = _make_calibrator(pipe_mlp, method='isotonic')
    calib_mlp.fit(X_val, y_val)
    mlp_time = time.time() - start
    y_pred_proba = calib_mlp.predict_proba(X_test)[:, 1]
    y_pred = (y_pred_proba >= 0.5).astype(int)
    results['neural_network'] = {**summarize_metrics(y_test, y_pred, y_pred_proba), 'train_time_seconds': mlp_time}
    joblib.dump(calib_mlp, OUT / 'model_neural_network_calibrated.joblib')

    # Save metrics
    with open(OUT / 'metrics_summary.json', 'w') as fh:
        json.dump(results, fh, indent=2)
    print('Saved metrics to', OUT / 'metrics_summary.json')

    return results


def summarize_metrics(y_true, y_pred, y_proba):
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    auc = float(roc_auc_score(y_true, y_proba)) if len(np.unique(y_true)) > 1 else None
    brier = float(brier_score_loss(y_true, y_proba))
    return {'classification_report': report, 'roc_auc': auc, 'brier_score': brier}


def main():
    path = find_latest_processed()
    print('Loading processed data from', path)
    df = load_data(path)
    X, y, numeric_feats, binary_feats, cat_feats = prepare_features(df)
    preprocessor = build_preprocessor(numeric_feats, binary_feats, cat_feats)
    results = train_and_evaluate(X, y, preprocessor)
    print('Training complete. Results saved to', OUT)


if __name__ == '__main__':
    main()
