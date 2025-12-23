"""Evaluation metrics and helpers."""
from sklearn.metrics import roc_auc_score, log_loss


def compute_metrics(y_true, y_pred_proba):
    return {
        'roc_auc': roc_auc_score(y_true, y_pred_proba),
        'log_loss': log_loss(y_true, y_pred_proba)
    }
