import matplotlib.pyplot as plt
import pandas as pd
import os

RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)


def plot_pauc_comp(
    temporal_results_df,
    same_day_pauc,
    model_name,
):
    """
    Plot same-day vs temporal pAUC@1%FPR for a given model.

    Parameters
    ----------
    temporal_results_df : pd.DataFrame
        Must contain columns: ['day', 'model', 'pauc@1%fpr']

    same_day_pauc : float
        pAUC@1%FPR from same-day 80/20 split (baseline)

    model_name : str
        Model identifier used in results_df (e.g., 'RF', 'XGB')
    """

    subset = temporal_results_df[
        temporal_results_df["model"] == model_name
    ].sort_values("day")

    days = subset["day"].values
    temporal_pauc = subset["pauc@1%fpr"].values

    plt.figure(figsize=(9, 5))

    # Same-day baseline (horizontal reference)
    plt.axhline(
        y=same_day_pauc,
        color="black",
        linestyle="-",
        linewidth=2,
        label="Same-Day Baseline (80/20)"
    )

    # Temporal degradation curve
    plt.plot(
        days,
        temporal_pauc,
        marker="o",
        linestyle="--",
        linewidth=2,
        label="Temporal Evaluation"
    )

    plt.xlabel("Test Day")
    plt.ylabel("pAUC@1% FPR")
    plt.title(
        f"{model_name} Ranking Stability Under Temporal Drift",
        pad=12
    )

    plt.xticks(rotation=45)
    plt.ylim(0, 1.05)
    plt.grid(alpha=0.3)
    plt.legend(frameon=True)

    plt.tight_layout()

    out_path = (
        f"{RESULTS_DIR}/{model_name.lower()}_pauc_same_day_vs_temporal.png"
    )
    plt.savefig(out_path, dpi=300)
    plt.show()
    plt.close()

    print(f"Saved: {out_path}")
