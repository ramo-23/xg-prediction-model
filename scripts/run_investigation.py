#!/usr/bin/env python3
"""Run a lightweight investigation: evaluate saved models and produce figures/metrics.
This script mirrors the logic in notebooks/04_investigation_and_tuning.ipynb
"""
from pathlib import Path
import json
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_auc_score,
    brier_score_loss,
    precision_recall_curve,
    average_precision_score,
    roc_curve,
    classification_report,
    confusion_matrix,
)
from sklearn.calibration import calibration_curve


def find_repo_root(start=Path.cwd(), markers=("setup.py", "requirements.txt", "README.md")):
    cur = start.resolve()
    for _ in range(10):
        if any((cur / m).exists() for m in markers):
            return cur
        if cur.parent == cur:
            break
        cur = cur.parent
    return start.resolve()


def main():
    repo_root = find_repo_root()
    proc_dir = repo_root / "data" / "processed"
    files = sorted(proc_dir.glob("processed_shots_*.csv"), key=lambda p: p.stat().st_mtime)
    if not files:
        raise SystemExit("No processed_shots_*.csv files in data/processed/")
    data_path = files[-1]
    print("Using processed CSV:", data_path)
    df = pd.read_csv(data_path)
    if "outcome" not in df.columns:
        raise SystemExit("processed data missing outcome column")
    y = df["outcome"].astype(str).str.lower().eq("goal").astype(int)

    numeric_feats = [c for c in ["distance", "minute_num"] if c in df.columns]
    binary_feats = [c for c in ["body_head", "body_foot", "body_other", "big_chance", "half"] if c in df.columns]
    cat_feats = [c for c in ["shot_type", "assist_type"] if c in df.columns]
    feature_cols = numeric_feats + binary_feats + cat_feats
    X = df[feature_cols].copy()
    print("Features used:", feature_cols)
    print("Class counts:", y.value_counts().to_dict())

    # splits
    from sklearn.model_selection import train_test_split

    X_train_full, X_hold, y_train_full, y_hold = train_test_split(X, y, test_size=0.4, stratify=y, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_hold, y_hold, test_size=0.5, stratify=y_hold, random_state=42)
    print("Train/Val/Test sizes:", len(X_train_full), len(X_val), len(X_test))

    model_files = {
        "logistic": repo_root / "results" / "metrics" / "model_logistic_calibrated.joblib",
        "random_forest": repo_root / "results" / "metrics" / "model_random_forest_calibrated.joblib",
        "xgboost": repo_root / "results" / "metrics" / "model_xgboost_calibrated.joblib",
        "neural_network": repo_root / "results" / "metrics" / "model_neural_network_calibrated.joblib",
    }

    eval_rows = []
    for name, path in model_files.items():
        if path.exists():
            print("Loading", name, path)
            mdl = joblib.load(path)
            try:
                proba = mdl.predict_proba(X_test)[:, 1]
            except Exception:
                try:
                    proba = mdl.predict_proba(X_test)[:, 1]
                except Exception:
                    print("predict_proba failed for", name)
                    continue
            pred = (proba >= 0.5).astype(int)
            auc_score = roc_auc_score(y_test, proba) if len(np.unique(y_test)) > 1 else None
            brier = brier_score_loss(y_test, proba)
            report = classification_report(y_test, pred, output_dict=True, zero_division=0)
            cm = confusion_matrix(y_test, pred)
            eval_rows.append({"model": name, "roc_auc": auc_score, "brier": brier, "proba": proba.tolist(), "report": report, "cm": cm.tolist()})
        else:
            print("Missing model file:", path)

    # create results dir
    out_dir = repo_root / "results" / "metrics"
    out_dir.mkdir(parents=True, exist_ok=True)

    # save metrics summary
    metrics_summary = {r["model"]: {"roc_auc": r["roc_auc"], "brier": r["brier"], "cm": r["cm"]} for r in eval_rows}
    metrics_path = out_dir / "metrics_summary.json"
    with open(metrics_path, "w") as fh:
        json.dump(metrics_summary, fh, indent=2)
    print("Saved metrics to", metrics_path)

    # produce figures if any evaluations
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    if eval_rows:
        # ROC
        plt.figure(figsize=(8, 6))
        for r in eval_rows:
            proba = np.array(r["proba"])
            fpr, tpr, _ = roc_curve(y_test, proba)
            plt.plot(fpr, tpr, label=f"{r['model']} (AUC={r['roc_auc']:.3f})")
        plt.plot([0, 1], [0, 1], "k--", alpha=0.3)
        plt.xlabel("FPR")
        plt.ylabel("TPR")
        plt.title("ROC curves (test set)")
        plt.legend(loc="lower right")
        plt.grid(True)
        roc_fig = fig_dir / "roc_curves.png"
        plt.savefig(roc_fig, dpi=150, bbox_inches="tight")
        plt.close()

        # Calibration + hist
        plt.figure(figsize=(8, 6))
        for r in eval_rows:
            proba = np.array(r["proba"])
            prob_true, prob_pred = calibration_curve(y_test, proba, n_bins=10)
            plt.plot(prob_pred, prob_true, marker="o", label=f"{r['model']} (Brier={r['brier']:.3f})")
        plt.plot([0, 1], [0, 1], "k--", alpha=0.5)
        plt.xlabel("Mean predicted probability")
        plt.ylabel("Fraction of positives")
        plt.title("Calibration plot (reliability diagram)")
        plt.legend()
        plt.grid(True)
        cal_fig = fig_dir / "calibration_reliability.png"
        plt.savefig(cal_fig, dpi=150, bbox_inches="tight")
        plt.close()

        plt.figure(figsize=(8, 3))
        for r in eval_rows:
            plt.hist(np.array(r["proba"]), bins=20, alpha=0.4, label=r["model"])
        plt.legend()
        plt.title("Predicted probability distribution (test set)")
        hist_fig = fig_dir / "probability_histograms.png"
        plt.savefig(hist_fig, dpi=150, bbox_inches="tight")
        plt.close()

        # PR curves
        plt.figure(figsize=(8, 6))
        for r in eval_rows:
            proba = np.array(r["proba"])
            precision, recall, _ = precision_recall_curve(y_test, proba)
            ap = average_precision_score(y_test, proba)
            plt.plot(recall, precision, label=f"{r['model']} (AP={ap:.3f})")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title("Precision-Recall curves (test set)")
        plt.legend()
        plt.grid(True)
        pr_fig = fig_dir / "precision_recall_curves.png"
        plt.savefig(pr_fig, dpi=150, bbox_inches="tight")
        plt.close()

        print("Saved figures to", fig_dir)
        # write a short interpretation summary to file
        report_lines = []
        report_lines.append('Model summary (roc_auc, brier):')
        for r in eval_rows:
            report_lines.append(f"- {r['model']}: ROC AUC={r['roc_auc']:.3f}, Brier={r['brier']:.4f}")
            report_lines.append(f"  Confusion matrix: {r.get('cm')}")
        report_lines.append('\nQuick interpretation:')
        for r in eval_rows:
            roc = r.get('roc_auc') or 0.0
            brier = r.get('brier') or 1.0
            if roc < 0.6:
                report_lines.append(f"- {r['model']}: Low discrimination (ROC AUC={roc:.3f}). Consider richer features or resampling/tuning.")
            elif roc < 0.7:
                report_lines.append(f"- {r['model']}: Moderate discrimination (ROC AUC={roc:.3f}). Consider calibration and threshold tuning.")
            else:
                report_lines.append(f"- {r['model']}: Good discrimination (ROC AUC={roc:.3f}). Evaluate calibration and thresholds.")
            if brier > 0.10:
                report_lines.append(f"  Brier score {brier:.3f} indicates noisy probability estimates; consider recalibration.")
        report_lines.append('\nSaved figures:')
        for p in sorted(fig_dir.glob('*.png')):
            report_lines.append(f"- {p.name}")
        int_path = out_dir / 'interpretation.txt'
        with open(int_path, 'w', encoding='utf-8') as fh:
            fh.write('\n'.join(report_lines))
        print('Wrote interpretation to', int_path)


if __name__ == "__main__":
    main()
