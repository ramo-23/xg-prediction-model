"""Model training wrappers."""
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb


def logistic_model(**kwargs):
    return LogisticRegression(**kwargs)


def random_forest_model(**kwargs):
    return RandomForestClassifier(**kwargs)


def xgboost_model(**kwargs):
    return xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', **kwargs)
