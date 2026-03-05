import numpy as np
from sklearn.metrics import precision_recall_curve, roc_curve, auc

# Precision–Recall AUC (Average Precision style)
def compute_pr_auc(y_true, y_scores):
    """
    Compute PR-AUC in a numerically stable and standard-compliant way.

    Note:
    - precision_recall_curve already returns recall in ascending order
    - reversing arrays is unnecessary and can introduce subtle errors
    """

    precision, recall, _ = precision_recall_curve(y_true, y_scores)

    # Ensure monotonic recall ordering
    if not np.all(np.diff(recall) >= 0):
        order = np.argsort(recall)
        recall = recall[order]
        precision = precision[order]

    return auc(recall, precision)

# Recall at fixed False Positive Rate (operational metric)
def recall_at_fpr(y_true, y_scores, target_fpr=0.01):
    """
    Compute recall (TPR) at or below a target false positive rate.

    This is an operationally meaningful metric for NIDS deployment
    and is stable across datasets and time.
    """

    fpr, tpr, thresholds = roc_curve(y_true, y_scores)

    # Identify valid operating points
    valid_idx = np.where(fpr <= target_fpr)[0]

    if len(valid_idx) == 0:
        # No feasible threshold meets FPR constraint
        return 0.0, None

    # Choose the point with highest recall under FPR constraint
    best_idx = valid_idx[np.argmax(tpr[valid_idx])]

    return float(tpr[best_idx]), float(thresholds[best_idx])

# Partial AUC at fixed False Positive Rate (operational metric)
def partial_auc_at_fpr(y_true, y_scores, max_fpr=0.01):
    """
    Compute partial AUC up to a maximum false positive rate.
    pAUC is normalized to [0, 1] by dividing by max_fpr.
    """

    fpr, tpr, _ = roc_curve(y_true, y_scores)

    # Restrict to low-FPR region
    mask = fpr <= max_fpr

    if mask.sum() < 2:
        return 0.0

    p_auc = auc(fpr[mask], tpr[mask])

    # Normalize so values are comparable across runs
    return p_auc / max_fpr