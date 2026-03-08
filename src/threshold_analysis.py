import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
import os

RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)


def threshold_analysis(
    model,
    X_test,
    y_test,
    model_name="Model",
    target_fpr=0.01
):
    """
    Select threshold that maximizes recall while keeping FPR <= target_fpr.
    This produces operationally stable thresholds suitable for temporal evaluation.
    """

    # Predicted probabilities
    y_probs = model.predict_proba(X_test)[:, 1]

    # ROC-derived operating points
    fpr, tpr, thresholds = roc_curve(y_test, y_probs)

    df = pd.DataFrame({
        "threshold": thresholds,
        "recall": tpr,
        "fpr": fpr
    })

    # Keep only feasible operating points
    feasible = df[df["fpr"] <= target_fpr]

    if feasible.empty:
        print(
            f"[WARN] {model_name}: No threshold satisfies "
            f"FPR ≤ {target_fpr:.2%}. Using lowest-FPR point."
        )
        best = df.loc[df["fpr"].idxmin()]
    else:
        # Maximize recall under FPR constraint
        best = feasible.loc[feasible["recall"].idxmax()]

    # Report selection
    print(f"\n=== {model_name} Threshold Selection ===")
    print(f"Target FPR:       {target_fpr:.2%}")
    print(f"Chosen threshold: {best['threshold']:.3f}")
    print(f"Recall @ FPR:     {best['recall']:.4f}")
    print(f"Actual FPR:       {best['fpr']:.4%}")

    # Plot Recall vs Threshold
    plt.figure(figsize=(6, 4))
    plt.plot(df["threshold"], df["recall"], marker="o", linewidth=1)
    plt.axvline(
        best["threshold"],
        linestyle="--",
        color="green",
        label=f"Chosen ({best['threshold']:.2f})"
    )
    plt.xlabel("Decision Threshold")
    plt.ylabel("Recall (TPR)")
    plt.title(f"{model_name}: Recall vs Threshold")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(
        f"{RESULTS_DIR}/{model_name.lower().replace(' ', '_')}_threshold_recall.png",
        dpi=300
    )
    plt.close()

    # Plot FPR vs Threshold
    plt.figure(figsize=(6, 4))
    plt.plot(df["threshold"], df["fpr"], marker="o", linewidth=1, color="red")
    plt.axhline(
        target_fpr,
        linestyle="--",
        color="gray",
        label=f"Target FPR ({target_fpr:.0%})"
    )
    plt.axvline(
        best["threshold"],
        linestyle="--",
        color="green",
        label=f"Chosen ({best['threshold']:.2f})"
    )
    plt.xlabel("Decision Threshold")
    plt.ylabel("False Positive Rate")
    plt.title(f"{model_name}: FPR vs Threshold")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(
        f"{RESULTS_DIR}/{model_name.lower().replace(' ', '_')}_threshold_fpr.png",
        dpi=300
    )
    plt.close()

    return best["threshold"], df
