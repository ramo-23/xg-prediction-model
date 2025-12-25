"""Model factory wrappers.

This module provides lightweight factory functions that return sklearn-/xgboost
estimators pre-configured with reasonable defaults used by the training scripts.

Keep these functions simple — they are intended as convenience helpers for
experimentation and training pipelines.
"""

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb


def logistic_model(**kwargs):
    """Return a `LogisticRegression` estimator.

    Any keyword args are forwarded to the constructor.
    """
    return LogisticRegression(**kwargs)


def random_forest_model(**kwargs):
    """Return a `RandomForestClassifier` estimator.

    Any keyword args are forwarded to the constructor.
    """
    return RandomForestClassifier(**kwargs)


def xgboost_model(**kwargs):
    """Return an `XGBClassifier` with sensible defaults for xG modeling.

    Disables the legacy label encoder and sets `eval_metric='logloss'` by
    default — additional args can be provided via `kwargs`.
    """
    return xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', **kwargs)
