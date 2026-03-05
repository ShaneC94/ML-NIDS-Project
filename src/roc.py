from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt
import numpy as np
import os

from src.metrics import partial_auc_at_fpr

RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)


def plot_roc_curve(
    model,
    X_test,
    y_test,
    model_name="Model",
    target_fpr=0.01,
):
    """
    Plot ROC curve with explicit emphasis on the low-FPR operating region.

    Design intent:
    - Full ROC shown for intuition only
    - pAUC@target_fpr is the ranking metric of interest
    - Visualization highlights the region that actually matters in deployment
    """

    # Predicted probabilities
    y_prob = model.predict_proba(X_test)[:, 1]

    # ROC computation
    fpr, tpr, thresholds = roc_curve(y_test, y_prob)

    # Partial AUC (ranking stability metric)
    p_auc = partial_auc_at_fpr(
        y_test,
        y_prob,
        max_fpr=target_fpr,
    )

    # Plot
    plt.figure(figsize=(6, 5))

    # Full ROC (context only)
    plt.plot(
        fpr,
        tpr,
        linewidth=2,
        label=f"{model_name} (pAUC@{int(target_fpr*100)}% = {p_auc:.4f})",
    )

    # Random baseline
    plt.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        color="gray",
        linewidth=1,
        label="Random Classifier",
    )

    # Highlight low-FPR region
    plt.axvline(
        target_fpr,
        linestyle="--",
        color="red",
        linewidth=1,
        label=f"Target FPR ({target_fpr:.0%})",
    )

    # Shade pAUC region
    mask = fpr <= target_fpr
    plt.fill_between(
        fpr[mask],
        tpr[mask],
        alpha=0.2,
        color="red",
        label="pAUC Region",
    )

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate (Recall)")
    plt.title(f"{model_name} ROC Curve (Low-FPR Focused)")

    # Zoom to meaningful region
    plt.xlim(0, min(0.1, max(fpr[mask]) * 1.2 if mask.any() else target_fpr))
    plt.ylim(0, 1.05)

    plt.grid(alpha=0.3)
    plt.legend(loc="lower right", frameon=True)

    # Save
    filename = f"{RESULTS_DIR}/{model_name.lower().replace(' ', '_')}_roc_pauc.png"
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.show()
    plt.close()

    return fpr, tpr, thresholds