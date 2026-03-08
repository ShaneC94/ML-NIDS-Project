from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    average_precision_score,
    roc_curve,
)
import numpy as np
import matplotlib.pyplot as plt
import os

from src.metrics import partial_auc_at_fpr

RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)


def evaluate_model(
    model,
    X_test,
    y_test,
    model_name="Model",
    threshold=0.5,
    target_fpr=0.01,
    plot=True,
):
    """
    Evaluate a probabilistic classifier using deployment-aligned metrics.

    Metrics philosophy:
    - pAUC@FPR is the ranking stability metric (PRIMARY)
    - Recall@FPR is the operational performance metric (PRIMARY)
    - PR-AUC, confusion matrix, and classification report are DIAGNOSTIC ONLY
    """

    # Predicted probabilities
    y_prob = model.predict_proba(X_test)[:, 1]

    print(f"\n=== {model_name} Evaluation ===")
    print(f"Input decision threshold (diagnostic only): {threshold:.3f}")

    # ROC-derived operating points
    fpr, tpr, thresholds = roc_curve(y_test, y_prob)
    valid_idx = np.where(fpr <= target_fpr)[0]

    if len(valid_idx) == 0:
        recall_at_fpr = 0.0
        threshold_at_fpr = None
    else:
        best_idx = valid_idx[np.argmax(tpr[valid_idx])]
        recall_at_fpr = float(tpr[best_idx])
        threshold_at_fpr = float(thresholds[best_idx])

    # Partial AUC @ fixed FPR (ranking stability)
    p_auc = partial_auc_at_fpr(
        y_test,
        y_prob,
        max_fpr=target_fpr,
    )

    print(f"pAUC@{int(target_fpr*100)}% FPR: {p_auc:.4f}")
    print(f"Recall@{int(target_fpr*100)}% FPR: {recall_at_fpr:.4f}")
    print(
        f"Operational threshold@{int(target_fpr*100)}% FPR: "
        f"{threshold_at_fpr if threshold_at_fpr is not None else 'N/A'}"
    )

    # Diagnostic predictions (classification report only)
    y_pred_diag = (y_prob >= threshold).astype(int)
    print("\n[Diagnostic] Classification report (fixed threshold):")
    print(classification_report(y_test, y_pred_diag, digits=4))

    # PR-AUC (diagnostic only)
    pr_auc = average_precision_score(y_test, y_prob)
    print(f"PR-AUC (diagnostic): {pr_auc:.4f}")

    # Confusion Matrix @ OPERATIONAL threshold
    if plot and threshold_at_fpr is not None:
        y_pred_operational = (y_prob >= threshold_at_fpr).astype(int)
        cm = confusion_matrix(y_test, y_pred_operational)

        plt.figure(figsize=(5, 4))
        plt.imshow(cm, cmap="Blues")
        plt.title(
            f"{model_name} Confusion Matrix\n"
            f"(Recall@{int(target_fpr*100)}% FPR Operating Point)"
        )
        plt.colorbar()

        classes = ["Benign", "Attack"]
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes)
        plt.yticks(tick_marks, classes)

        thresh = cm.max() / 2
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(
                    j,
                    i,
                    f"{cm[i, j]:,}",
                    ha="center",
                    va="center",
                    color="white" if cm[i, j] > thresh else "black",
                )

        plt.ylabel("True Label")
        plt.xlabel("Predicted Label")
        plt.tight_layout()

        plt.savefig(
            f"{RESULTS_DIR}/{model_name.lower().replace(' ', '_')}_confusion_matrix_operational.png",
            dpi=300,
        )
        plt.close()

    # Precision–Recall Curve (diagnostic visualization)
    if plot:
        precision, recall, _ = precision_recall_curve(y_test, y_prob)

        plt.figure(figsize=(6, 4))
        plt.plot(recall, precision)
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title(f"{model_name} Precision–Recall Curve (Diagnostic)")
        plt.grid(alpha=0.3)
        plt.tight_layout()

        plt.savefig(
            f"{RESULTS_DIR}/{model_name.lower().replace(' ', '_')}_pr_curve.png",
            dpi=300,
        )
        plt.close()

    # Return metrics for tables / CSVs
    return {
        f"pauc@{int(target_fpr*100)}%fpr": p_auc,
        "pr_auc": pr_auc,
        f"recall@{int(target_fpr*100)}%fpr": recall_at_fpr,
        f"threshold@{int(target_fpr*100)}%fpr": threshold_at_fpr,
    }