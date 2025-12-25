"""Evaluation metrics and helpers.

Provides small wrappers used by training and reporting scripts. Functions here
expect NumPy arrays or pandas Series for true labels and predicted
probabilities.
"""

from sklearn.metrics import roc_auc_score, log_loss


def compute_metrics(y_true, y_pred_proba):
    """Compute a small set of evaluation metrics.

    Args:
        y_true: Array-like ground truth binary labels (0/1).
        y_pred_proba: Array-like predicted positive-class probabilities.

    Returns:
        dict with `roc_auc` and `log_loss` values.
    """
    return {
        'roc_auc': roc_auc_score(y_true, y_pred_proba),
        'log_loss': log_loss(y_true, y_pred_proba)
    }
