# Expected Goals (xG) Model

## Overview
This repository implements an end-to-end expected goals (xG) modelling pipeline, covering data preparation, feature engineering, model training, evaluation, calibration, and visualization. The project is fully reproducible and designed to be easy to run, inspect, and extend.

## Why this project?
This project began as a way to apply concepts from my final-year studies to a domain I genuinely enjoy: football analytics. The goal is both educational and practical — to build an interpretable xG model, understand which features influence goal probability, and demonstrate sound model evaluation practices such as calibration and discrimination.

If you are looking for a hands-on example of applied machine learning in sports analytics — with code you can run, modify, and build upon — this repository is intended for you.

## Who this is for
- Students and hobbyist data scientists learning how to structure end-to-end ML projects.
- Football analysts interested in how xG models are built, evaluated, and interpreted.

## Repository structure
- `data/processed/`  
  Processed tabular shot-event CSV files used for modelling.

- `src/`  
  Core modules for data collection, preprocessing, feature engineering, model training, and evaluation utilities.

- `scripts/`  
  Runnable scripts for training models and running experiments:
  - `train_models.py`
  - `run_investigation.py`
  - `run_imbalance_experiments.py`

- `notebooks/`  
  Reproducible notebooks for exploratory data analysis (EDA), feature engineering, training, evaluation, and visualization.

- `results/metrics/`  
  Saved model artifacts (`.joblib`), metrics JSON files, figures, and CSV summaries.

## Quickstart

### 1. Create and activate a virtual environment (Windows example)
```powershell
python -m venv venv
& venv\Scripts\Activate.ps1
pip install -r requirements.txt
````

### 2. Train models

This step trains and calibrates models and writes evaluation outputs to `results/metrics/metrics_summary.json`.

```powershell
python scripts/train_models.py
```

### 3. Run the investigation

Generates figures and a written interpretation in `results/metrics/interpretation.txt`.

```powershell
python scripts/run_investigation.py
```

### 4. (Optional) Run class imbalance experiments

Requires `imblearn` for SMOTE-based experiments.

```powershell
python scripts/run_imbalance_experiments.py
```

## Notes

* Notebooks assume the latest processed CSV is located in `data/processed/` and follows the naming pattern `processed_shots_*.csv`.
* Calibrated models are saved in `results/metrics/` using filenames such as `model_<name>_calibrated.joblib`.
* The notebook `notebooks/05_visualization_and_dashboard.ipynb` provides lightweight interactive analysis, including:

  * Model comparison
  * Feature importance extraction
  * Player and team xG aggregation
  * A single-shot probability predictor using `ipywidgets`
    (`ipywidgets` is imported dynamically to avoid editor missing-import warnings.)
* A previous Streamlit starter app has been removed in favor of notebook-based interactive exploration.

## Developer notes

- Run tests:

```powershell
& venv\Scripts\Activate.ps1
python -m pytest -q
```

- Code layout: see `src/` for core modules. Functions are intentionally small
  and tested via `tests/` for easier maintenance.

- When adding features or changing inference logic, update or add unit tests
  in `tests/test_features.py` to cover edge cases and expected behaviors.

- Formatting: follow standard Python conventions (PEP8). Use `black` or
  `ruff` if available in your editor/CI.

## Contributing

Contributions are welcome. If you have ideas for additional visualizations, feature enhancements, model tuning, or deployment helpers, feel free to open an issue or submit a focused pull request.

## Contact

Maintainer: **Ramokhele Manyeli**

```
```
