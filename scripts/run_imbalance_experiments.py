#!/usr/bin/env python3
"""Run simple imbalance-handling experiments and save results.
Experiments:
 - LogisticRegression with class_weight='balanced'
 - LogisticRegression with upsampling
 - LogisticRegression with downsampling
 - (optional) LogisticRegression with SMOTE if imblearn installed

Saves JSON results under results/metrics/experiments/
"""
from pathlib import Path
import json
import pandas as pd
import numpy as np
import joblib

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, brier_score_loss, classification_report, confusion_matrix, average_precision_score
from sklearn.model_selection import train_test_split
from sklearn.utils import resample


def find_repo_root(start=Path.cwd(), markers=("setup.py", "requirements.txt", "README.md")):
    cur = start.resolve()
    for _ in range(10):
        if any((cur / m).exists() for m in markers):
            return cur
        if cur.parent == cur:
            break
        cur = cur.parent
    return start.resolve()


def load_data(repo_root):
    proc_dir = repo_root / "data" / "processed"
    files = sorted(proc_dir.glob("processed_shots_*.csv"), key=lambda p: p.stat().st_mtime)
    if not files:
        raise SystemExit("No processed_shots_*.csv files in data/processed/")
    df = pd.read_csv(files[-1])
    if "outcome" not in df.columns:
        raise SystemExit("processed data missing outcome column")
    y = df["outcome"].astype(str).str.lower().eq("goal").astype(int)
    numeric_feats = [c for c in ["distance", "minute_num"] if c in df.columns]
    binary_feats = [c for c in ["body_head", "body_foot", "body_other", "big_chance", "half"] if c in df.columns]
    cat_feats = [c for c in ["shot_type", "assist_type"] if c in df.columns]
    feature_cols = numeric_feats + binary_feats + cat_feats
    X = df[feature_cols].copy()
    return X, y, numeric_feats, binary_feats, cat_feats


def make_preprocessor(numeric_feats, binary_feats, cat_feats):
    num_pipe = Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]) if numeric_feats else None
    bin_pipe = Pipeline([("imp", SimpleImputer(strategy="most_frequent"))]) if binary_feats else None
    cat_pipe = Pipeline([("imp", SimpleImputer(strategy="constant", fill_value="missing")), ("ohe", OneHotEncoder(handle_unknown="ignore"))]) if cat_feats else None
    transformers = []
    if numeric_feats: transformers.append(("num", num_pipe, numeric_feats))
    if binary_feats: transformers.append(("bin", bin_pipe, binary_feats))
    if cat_feats: transformers.append(("cat", cat_pipe, cat_feats))
    preprocessor = ColumnTransformer(transformers, remainder="drop")
    return preprocessor


def evaluate_pipe(pipe, X_test, y_test):
    proba = pipe.predict_proba(X_test)[:,1]
    pred = (proba >= 0.5).astype(int)
    return {
        "roc_auc": float(roc_auc_score(y_test, proba)),
        "brier": float(brier_score_loss(y_test, proba)),
        "ap": float(average_precision_score(y_test, proba)),
        "cm": confusion_matrix(y_test, pred).tolist(),
        "report": classification_report(y_test, pred, output_dict=True, zero_division=0),
    }


def main():
    repo_root = find_repo_root()
    X, y, numeric_feats, binary_feats, cat_feats = load_data(repo_root)
    preprocessor = make_preprocessor(numeric_feats, binary_feats, cat_feats)

    X_train_full, X_hold, y_train_full, y_hold = train_test_split(X, y, test_size=0.4, stratify=y, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_hold, y_hold, test_size=0.5, stratify=y_hold, random_state=42)

    out_dir = repo_root / "results" / "metrics" / "experiments"
    out_dir.mkdir(parents=True, exist_ok=True)

    experiments = {}

    # 1) class_weight balanced logistic
    pipe = Pipeline([("pre", preprocessor), ("clf", LogisticRegression(max_iter=1000, class_weight="balanced"))])
    pipe.fit(X_train_full, y_train_full)
    experiments["logistic_class_weight_balanced"] = evaluate_pipe(pipe, X_test, y_test)

    # 2) upsample positives
    train_df = pd.concat([X_train_full, y_train_full.rename("y")], axis=1)
    minor = train_df[train_df["y"]==1]
    major = train_df[train_df["y"]==0]
    if len(minor)>0:
        minor_up = resample(minor, replace=True, n_samples=len(major), random_state=42)
        train_up = pd.concat([major, minor_up])
        X_train_up = train_up[train_up.columns.difference(["y"])].loc[:, train_up.columns.difference(["y"])].copy()
        # ensure columns order
        X_train_up = train_up[feature_cols]
        y_train_up = train_up["y"]
        pipe_up = Pipeline([("pre", preprocessor), ("clf", LogisticRegression(max_iter=1000))])
        pipe_up.fit(X_train_up, y_train_up)
        experiments["logistic_upsample"] = evaluate_pipe(pipe_up, X_test, y_test)

    # 3) downsample negatives
    if len(major)>0 and len(minor)>0:
        major_down = resample(major, replace=False, n_samples=len(minor), random_state=42)
        train_down = pd.concat([major_down, minor])
        X_train_down = train_down[feature_cols]
        y_train_down = train_down["y"]
        pipe_down = Pipeline([("pre", preprocessor), ("clf", LogisticRegression(max_iter=1000))])
        pipe_down.fit(X_train_down, y_train_down)
        experiments["logistic_downsample"] = evaluate_pipe(pipe_down, X_test, y_test)

    # 4) optional SMOTE if available
    # optional SMOTE via dynamic import (avoids static import error in editors without imblearn)
    try:
        import importlib
        smote_mod = importlib.import_module("imblearn.over_sampling")
        SMOTE = getattr(smote_mod, "SMOTE")
        smote = SMOTE(random_state=42)
        X_res, y_res = smote.fit_resample(X_train_full, y_train_full)
        pipe_sm = Pipeline([("pre", preprocessor), ("clf", LogisticRegression(max_iter=1000))])
        pipe_sm.fit(X_res, y_res)
        experiments["logistic_smote"] = evaluate_pipe(pipe_sm, X_test, y_test)
    except Exception:
        experiments["logistic_smote"] = {"error": "imblearn not installed or SMOTE failed"}

    # save experiments
    with open(out_dir / "imbalance_experiments_summary.json", "w", encoding="utf-8") as fh:
        json.dump(experiments, fh, indent=2)

    print('Saved experiment summary to', out_dir / 'imbalance_experiments_summary.json')


if __name__ == "__main__":
    main()
