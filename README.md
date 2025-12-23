# Expected Goals (xG) Model
Overview
--------
This repository contains data preparation, feature engineering, model training, evaluation, and visualization artifacts for an expected-goals (xG) prediction workflow.

Why this project
-----------------
This project started as a way to apply concepts learned during my final-year studies to something I love — football. The goal is both practical and educational: build an interpretable xG model, explore which features drive probability estimates, and provide clear, reproducible analysis so others can learn from the pipeline. If you want a hands-on example of applied ML in sport analytics — with code you can run, tweak, and extend — this repo is for you.

Who this is for
---------------
- Students and hobbyist data scientists learning end-to-end ML pipelines.
- Football analysts curious about what contributes to xG and how to evaluate calibration and discrimination.
- Developers looking for a reproducible baseline to extend with richer features or deployment.

What's included
---------------
- `data/processed/`: processed tabular shot-event CSVs used for modeling.
- `src/`: data collection, preprocessing, feature engineering, model utilities.
- `scripts/`: runnable scripts to train models, run investigations, and run imbalance experiments (see `scripts/train_models.py`, `scripts/run_investigation.py`, `scripts/run_imbalance_experiments.py`).
- `notebooks/`: analysis and reproducible notebooks (EDA, feature engineering, training, evaluation, visualization).
- `results/metrics/`: saved model artifacts (`.joblib`), metrics JSON, figures, and CSV summaries.

Quickstart
----------
1. Create and activate a Python virtual environment (Windows example):

```powershell
python -m venv venv
& venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

2. Train models (writes calibrated models and `results/metrics/metrics_summary.json`):

```powershell
python scripts/train_models.py
```

3. Run the scripted investigation (generates figures and `results/metrics/interpretation.txt`):

```powershell
python scripts/run_investigation.py
```

4. Run imbalance experiments (optional — requires `imblearn` for SMOTE):

```powershell
python scripts/run_imbalance_experiments.py
```

Notes
-----
- Notebooks assume the latest processed CSV is in `data/processed/` with filename matching `processed_shots_*.csv`.
- Calibrated models are saved in `results/metrics/` with names like `model_<name>_calibrated.joblib`.
- A lightweight interactive visualization notebook `notebooks/05_visualization_and_dashboard.ipynb` provides model comparison, feature importance extraction, player/team xG aggregation, and an ipywidgets-based single-shot predictor (ipywidgets is imported dynamically to avoid editor missing-import diagnostics).
- The previous Streamlit starter app (`app/streamlit_app.py`) has been removed; see the notebooks for interactive exploration.

Contributing
------------
If you'd like additional visualizations, model tuning, or deployment helpers, open an issue or send a PR with a focused change.

Contact
-------
Project maintainer: Ramokhele Manyeli.

